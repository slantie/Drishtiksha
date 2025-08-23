# src/ml/models/siglip_lstm_detector.py

import os
import time
import torch
import logging
import matplotlib
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
from src.ml.base import BaseModel
from transformers import AutoProcessor
from src.ml.utils import extract_frames
from typing import Any, Dict, Tuple, Optional, List
from src.ml.event_publisher import publish_progress
from src.ml.architectures.siglip_lstm_legacy import create_legacy_lstm_model
from src.ml.architectures.siglip_lstm import create_lstm_model as create_v4_model
from src.config import SiglipLSTMv1Config, SiglipLSTMv3Config, SiglipLSTMv4Config

# Add matplotlib import here for the visualization method
matplotlib.use("Agg")

logger = logging.getLogger(__name__)
class SiglipLSTMV1(BaseModel):
    config: SiglipLSTMv1Config

    def load(self) -> None:
        start_time = time.time()
        try:
            model_architecture_config = self.config.model_definition.model_dump()
            self.model = create_legacy_lstm_model(model_architecture_config)
            state_dict = torch.load(self.config.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(self.config.processor_path, use_fast=True)
            load_time = time.time() - start_time
            logger.info(
                f"✅ Loaded Model: '{self.config.class_name}'\t | Device: '{self.device}'\t | Time: {load_time:.2f} seconds."
            )
        except Exception as e:
            logger.error(f"Failed to load model '{self.config.class_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model '{self.config.class_name}'") from e

    def predict(self, video_path: str, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        frames = extract_frames(video_path, self.config.num_frames)
        if not frames:
            raise ValueError(f"Could not extract frames from the video: {video_path}")
        inputs = self.processor(images=frames, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        with torch.no_grad():
            logits = self.model(pixel_values, num_frames_per_video=self.config.num_frames)
            prob_fake = torch.sigmoid(logits.squeeze()).item()
        predicted_class_id = 1 if prob_fake > 0.5 else 0
        confidence = prob_fake if predicted_class_id == 1 else 1 - prob_fake
        processing_time = time.time() - start_time
        label_map = {0: "REAL", 1: "FAKE"}
        return {
            "prediction": label_map[predicted_class_id],
            "confidence": confidence,
            "processing_time": processing_time,
        }
class SiglipLSTMV3(SiglipLSTMV1):
    """
    Enhanced SIGLIP-LSTM V3 model. Inherits from V1 but overrides all prediction
    methods to provide advanced, detailed analysis capabilities.
    """
    config: SiglipLSTMv3Config

    def __init__(self, config: SiglipLSTMv3Config):
        super().__init__(config)
        self._last_detailed_result: Optional[Dict[str, Any]] = None
        self._last_video_path: Optional[str] = None

    def _analyze_video_frames(
        self, video_path: str, video_id: str = None, user_id: str = None
    ) -> Tuple[List[float], List[Image.Image]]:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file for processing: {video_path}")

        all_frames_pil = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                all_frames_pil.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        finally:
            cap.release()
            
        if not all_frames_pil:
            return [], []

        # --- REFACTORED: Use 50 fixed windows instead of per-frame analysis ---
        num_windows = 50
        total_frames = len(all_frames_pil)
        frame_scores = []
        seq_len = self.config.num_frames

        # Generate 50 evenly spaced anchor points to end our windows
        end_frame_indices = np.linspace(0, total_frames - 1, num_windows, dtype=int)

        for i, end_index in enumerate(tqdm(end_frame_indices, desc=f"Analyzing {num_windows} windows for {self.config.class_name}")):
            start_index = max(0, end_index - seq_len + 1)
            frame_window = all_frames_pil[start_index : end_index + 1]

            if len(frame_window) < seq_len:
                padding = [frame_window[0]] * (seq_len - len(frame_window))
                frame_window = padding + frame_window

            inputs = self.processor(images=frame_window, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(self.device)

            with torch.no_grad():
                logits = self.model(pixel_values, num_frames_per_video=seq_len)
                prob_fake = torch.sigmoid(logits.squeeze()).item()
            
            frame_scores.append(prob_fake)
            
            if video_id and user_id:
                publish_progress({
                    "videoId": video_id, "userId": user_id, "event": "FRAME_ANALYSIS_PROGRESS",
                    "message": f"Processed window {i + 1}/{num_windows}",
                    "data": {"modelName": self.config.class_name, "progress": i + 1, "total": num_windows},
                })
        # --- END REFACTOR ---

        return frame_scores, all_frames_pil

    def _get_or_run_detailed_analysis(self, video_path: str, **kwargs) -> Dict[str, Any]:
        """
        Runs a two-stage analysis: first, a detailed per-frame analysis, and second,
        a final authoritative prediction based on a representative sequence of frames.
        Caches the result to avoid re-computation for the same video.
        """
        if self._last_video_path == video_path and self._last_detailed_result:
            logger.info(f"✅ Using cached detailed analysis for {os.path.basename(video_path)}")
            return self._last_detailed_result

        video_id = kwargs.get("video_id")
        user_id = kwargs.get("user_id")
        start_time = time.time()

        frame_scores, all_frames_pil = self._analyze_video_frames(
            video_path, video_id=video_id, user_id=user_id
        )

        if not frame_scores or not all_frames_pil:
            raise ValueError("Frame analysis returned no scores or frames.")

        frame_indices = np.linspace(0, len(all_frames_pil) - 1, self.config.num_frames, dtype=int)
        final_frames_for_pred = [all_frames_pil[i] for i in frame_indices]
        video_inputs = self.processor(images=final_frames_for_pred, return_tensors="pt")
        pixel_values = video_inputs['pixel_values'].to(self.device)
        
        with torch.no_grad():
            logits_video = self.model(pixel_values, num_frames_per_video=self.config.num_frames)
            prob_fake_video = torch.sigmoid(logits_video.squeeze()).item()

        predicted_class_id = 1 if prob_fake_video > 0.5 else 0
        confidence = prob_fake_video if predicted_class_id == 1 else 1 - prob_fake_video
        label_map = {0: "REAL", 1: "FAKE"}
        authoritative_prediction = label_map[predicted_class_id]

        if video_id and user_id:
            publish_progress({
                "videoId": video_id, "userId": user_id, "event": "FRAME_ANALYSIS_PROGRESS",
                "message": f"Completed frame analysis for {self.config.class_name}",
                "data": {"modelName": self.config.class_name, "progress": len(frame_scores), "total": len(frame_scores)},
            })
        
        rolling_avg_scores = []
        rolling_window = deque(maxlen=self.config.rolling_window_size)
        for score in frame_scores:
            rolling_window.append(score)
            rolling_avg_scores.append(np.mean(list(rolling_window)))

        result = {
            "prediction": authoritative_prediction,
            "confidence": confidence,
            "processing_time": time.time() - start_time,
            "metrics": {
                "frame_count": len(frame_scores),
                "per_frame_scores": frame_scores,
                "rolling_average_scores": rolling_avg_scores,
                "final_average_score": np.mean(frame_scores),
                "max_score": max(frame_scores),
                "min_score": min(frame_scores),
                "score_variance": np.var(frame_scores),
                "suspicious_frames_count": sum(1 for s in frame_scores if s > 0.5),
            },
        }

        self._last_video_path = video_path
        self._last_detailed_result = result
        return result

    def predict_detailed(self, video_path: str, **kwargs) -> Dict[str, Any]:
        return self._get_or_run_detailed_analysis(video_path, **kwargs)

    def predict_frames(self, video_path: str, **kwargs) -> Dict[str, Any]:
        detailed_result = self._get_or_run_detailed_analysis(video_path, **kwargs)
        metrics = detailed_result["metrics"]
        frame_predictions = [
            {"frame_index": i, "score": score, "prediction": "FAKE" if score > 0.5 else "REAL"}
            for i, score in enumerate(metrics["per_frame_scores"])
        ]
        return {
            "overall_prediction": detailed_result["prediction"],
            "overall_confidence": detailed_result["confidence"],
            "processing_time": detailed_result["processing_time"],
            "frame_predictions": frame_predictions,
            "temporal_analysis": {
                "rolling_averages": metrics["rolling_average_scores"],
                "consistency_score": 1.0 - metrics["score_variance"],
            },
        }

    def predict_visual(self, video_path: str, **kwargs) -> str:
        import cv2
        import tempfile

        detailed_result = self._get_or_run_detailed_analysis(video_path, **kwargs)
        frame_scores = detailed_result["metrics"]["per_frame_scores"]

        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        output_temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        output_path = output_temp_file.name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        cap_frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            cap_frames.append(frame)
        cap.release()

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(6, 2.5))

        try:
            for i, score in enumerate(tqdm(frame_scores, desc="Generating visualization")):
                if i >= len(cap_frames): break
                frame = cap_frames[i].copy()

                # Generate and overlay graph
                ax.clear()
                ax.fill_between(range(i + 1), frame_scores[: i + 1], color="#FF4136", alpha=0.4)
                ax.plot(range(i + 1), frame_scores[: i + 1], color="#FF851B", linewidth=2)
                ax.axhline(y=0.5, color="white", linestyle="--", alpha=0.7)
                ax.set_ylim(-0.05, 1.05); ax.set_xlim(0, len(frame_scores))
                ax.set_title("Suspicion Analysis", fontsize=10); fig.tight_layout(pad=1.5)
                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                plot_img_rgba = np.frombuffer(buf, dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
                plot_img_bgr = cv2.cvtColor(plot_img_rgba, cv2.COLOR_RGBA2BGR)
                
                plot_h, plot_w, _ = plot_img_bgr.shape
                new_plot_h = int(frame_height * 0.3)
                new_plot_w = int(new_plot_h * (plot_w / plot_h))
                # Ensure overlay fits within frame
                if new_plot_w > frame_width:
                    new_plot_w = frame_width - 20 if frame_width > 20 else frame_width
                    new_plot_h = int(new_plot_w * (plot_h / plot_w))
                if new_plot_h > frame_height:
                    new_plot_h = frame_height - 20 if frame_height > 20 else frame_height
                    new_plot_w = int(new_plot_h * (plot_w / plot_h))
                resized_plot = cv2.resize(plot_img_bgr, (new_plot_w, new_plot_h))
                y_offset = 10
                x_offset = frame_width - new_plot_w - 10
                # Ensure offsets are valid
                if x_offset < 0:
                    x_offset = 0
                if y_offset + new_plot_h > frame_height:
                    y_offset = frame_height - new_plot_h
                # Only overlay if it fits
                if (y_offset >= 0 and x_offset >= 0 and
                    y_offset + new_plot_h <= frame_height and
                    x_offset + new_plot_w <= frame_width):
                    frame[y_offset:y_offset + new_plot_h, x_offset:x_offset + new_plot_w] = resized_plot
                # else: skip overlay if it doesn't fit

                # Generate gradient color bar
                green, yellow, red = np.array([0, 255, 0]), np.array([0, 255, 255]), np.array([0, 0, 255])
                interp = score * 2 if score < 0.5 else (score - 0.5) * 2
                color_np = green * (1 - interp) + yellow * interp if score < 0.5 else yellow * (1 - interp) + red * interp
                color = tuple(map(int, color_np))
                
                bar_height = 40
                cv2.rectangle(frame, (0, frame_height - bar_height), (frame_width, frame_height), color, -1)
                cv2.putText(frame, f"Live Frame Suspicion: {score:.2f}", (15, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                out.write(frame)
        finally:
            out.release()
            plt.close(fig)

        self._last_detailed_result = None
        self._last_video_path = None
        return output_path
class SiglipLSTMV4(SiglipLSTMV3):
    """
    Represents the V4 version of the Siglip-LSTM model, which incorporates
    a deeper classifier head with dropout for enhanced regularization.
    Inherits most of its prediction logic from V3.
    """
    config: SiglipLSTMv4Config

    def __init__(self, config: SiglipLSTMv4Config):
        super(SiglipLSTMV1, self).__init__(config)
        self._last_detailed_result: Optional[Dict[str, Any]] = None
        self._last_video_path: Optional[str] = None
        
    # Overload the load method for V4 to use the new architecture
    def load(self) -> None:
        start_time = time.time()
        try:
            model_architecture_config = self.config.model_definition.model_dump()
            self.model = create_v4_model(model_architecture_config)
            state_dict = torch.load(self.config.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(self.config.processor_path, use_fast=True)
            load_time = time.time() - start_time
            logger.info(
                f"✅ Loaded Model: '{self.config.class_name}'\t | Device: '{self.device}'\t | Time: {load_time:.2f} seconds."
            )
        except Exception as e:
            logger.error(f"Failed to load model '{self.config.class_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model '{self.config.class_name}'") from e