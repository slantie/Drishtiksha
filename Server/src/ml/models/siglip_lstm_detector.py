# src/ml/models/siglip_lstm_detector.py

import os
import cv2
import time
import torch
import logging
import tempfile
import matplotlib
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
from src.ml.base import BaseModel
from transformers import AutoProcessor
from src.ml.utils import extract_frames
from typing import Any, Dict, Optional, List, Tuple
from src.ml.event_publisher import publish_progress
from src.ml.architectures.siglip_lstm import create_siglip_lstm_model
from src.config import SiglipLSTMv1Config, SiglipLSTMv3Config, SiglipLSTMv4Config

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

class BaseSiglipLSTMDetector(BaseModel):
    """Base class for all SigLIP+LSTM models, containing shared logic."""

    def __init__(self, config):
        super().__init__(config)
        # This cache is simple and effective for the common case of a user requesting
        # multiple analysis types (detailed, frames, visual) for the same video in short succession.
        self._last_detailed_result: Optional[Dict[str, Any]] = None
        self._last_video_path: Optional[str] = None

    def predict(self, video_path: str, **kwargs) -> Dict[str, Any]:
        """Performs a quick, final prediction on a representative sequence of frames."""
        start_time = time.time()
        
        # FIX: Convert the generator from extract_frames into a list.
        frames = list(extract_frames(video_path, self.config.num_frames))
        
        if not frames:
            logger.error(f"Could not extract any frames from video: {video_path}")
            return {"prediction": "REAL", "confidence": 0.51, "processing_time": time.time() - start_time, "note": "Could not extract frames."}

        inputs = self.processor(images=frames, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        
        with torch.no_grad():
            logits = self.model(pixel_values, num_frames_per_video=self.config.num_frames)
            prob_fake = torch.sigmoid(logits.squeeze()).item()
            
        prediction = "FAKE" if prob_fake > 0.5 else "REAL"
        confidence = prob_fake if prediction == "FAKE" else 1 - prob_fake
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "processing_time": time.time() - start_time,
        }

    def _analyze_video_windows(
        self, video_path: str, video_id: str = None, user_id: str = None
    ) -> Tuple[List[float], int]:
        """Analyzes the video in 50 overlapping windows for detailed frame scores."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if total_frames == 0:
            return [], 0

        num_windows = 50
        frame_scores = []
        seq_len = self.config.num_frames
        end_frame_indices = np.linspace(seq_len - 1, total_frames - 1, num_windows, dtype=int)

        for i, end_index in enumerate(tqdm(end_frame_indices, desc=f"Analyzing {num_windows} windows for {self.config.class_name}")):
            start_index = max(0, end_index - seq_len + 1)
            indices_to_extract = np.linspace(start_index, end_index, seq_len, dtype=int).tolist()
            
            # FIX: Convert the generator to a list before passing to the processor.
            frame_window = list(extract_frames(video_path, num_frames=seq_len, specific_indices=indices_to_extract))

            if not frame_window:
                continue

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
        
        return frame_scores, total_frames

    def _get_or_run_detailed_analysis(self, video_path: str, **kwargs) -> Dict[str, Any]:
        if self._last_video_path == video_path and self._last_detailed_result:
            logger.info(f"✅ Using cached detailed analysis for {os.path.basename(video_path)}")
            return self._last_detailed_result

        start_time = time.time()
        video_id = kwargs.get("video_id")
        user_id = kwargs.get("user_id")

        frame_scores, total_frames = self._analyze_video_windows(video_path, video_id, user_id)
        if not frame_scores:
            raise ValueError("Frame analysis returned no scores.")

        # The final authoritative prediction uses the simpler, robust `predict` method.
        final_prediction_result = self.predict(video_path)

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
            "prediction": final_prediction_result["prediction"],
            "confidence": final_prediction_result["confidence"],
            "processing_time": time.time() - start_time,
            "metrics": {
                "frame_count": total_frames,
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
        detailed_result = self._get_or_run_detailed_analysis(video_path, **kwargs)
        frame_scores = detailed_result["metrics"]["per_frame_scores"]

        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        output_path = output_temp_file.name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(6, 2.5))

        try:
            for i in tqdm(range(total_frames), desc="Generating visualization"):
                ret, frame = cap.read()
                if not ret: break

                current_score_index = min(len(frame_scores) - 1, int((i / total_frames) * len(frame_scores)))
                score = frame_scores[current_score_index]

                # ... (rest of the visualization plotting logic is identical and remains here) ...
                ax.clear()
                ax.fill_between(range(len(frame_scores)), frame_scores, color="#FF4136", alpha=0.4)
                ax.plot(range(len(frame_scores)), frame_scores, color="#FF851B", linewidth=2)
                ax.plot(current_score_index, score, "o", color="yellow", markersize=8)
                ax.axhline(y=0.5, color="white", linestyle="--", alpha=0.7)
                ax.set_ylim(-0.05, 1.05); ax.set_xlim(0, len(frame_scores) -1 if len(frame_scores) > 1 else 1)
                ax.set_title("Suspicion Analysis", fontsize=10); fig.tight_layout(pad=1.5)
                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                plot_img_rgba = np.frombuffer(buf, dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
                plot_img_bgr = cv2.cvtColor(plot_img_rgba, cv2.COLOR_RGBA2BGR)
                
                plot_h, plot_w, _ = plot_img_bgr.shape
                new_plot_h = int(frame_height * 0.3)
                new_plot_w = int(new_plot_h * (plot_w / plot_h))
                resized_plot = cv2.resize(plot_img_bgr, (new_plot_w, new_plot_h))
                
                y_offset, x_offset = 10, frame_width - new_plot_w - 10
                frame[y_offset:y_offset + new_plot_h, x_offset:x_offset + new_plot_w] = resized_plot

                green, yellow, red = np.array([0, 255, 0]), np.array([0, 255, 255]), np.array([0, 0, 255])
                interp = score * 2 if score < 0.5 else (score - 0.5) * 2
                color_np = green * (1 - interp) + yellow * interp if score < 0.5 else yellow * (1 - interp) + red * interp
                color = tuple(map(int, color_np))
                
                bar_height = 40
                cv2.rectangle(frame, (0, frame_height - bar_height), (frame_width, frame_height), color, -1)
                cv2.putText(frame, f"Live Frame Suspicion: {score:.2f}", (15, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                out.write(frame)
        finally:
            cap.release()
            out.release()
            plt.close(fig)

        self._last_detailed_result = None
        self._last_video_path = None
        return output_path

class SiglipLSTMV1(BaseSiglipLSTMDetector):
    config: SiglipLSTMv1Config

    def load(self) -> None:
        start_time = time.time()
        try:
            model_architecture_config = self.config.model_definition.model_dump()
            self.model = create_siglip_lstm_model(model_architecture_config)
            state_dict = torch.load(self.config.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(self.config.processor_path)
            logger.info(f"✅ Loaded Model: '{self.config.class_name}' | Device: '{self.device}' | Time: {time.time() - start_time:.2f}s.")
        except Exception as e:
            logger.error(f"Failed to load model '{self.config.class_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model '{self.config.class_name}'") from e

class SiglipLSTMV3(BaseSiglipLSTMDetector):
    config: SiglipLSTMv3Config

    def load(self) -> None:
        start_time = time.time()
        try:
            model_architecture_config = self.config.model_definition.model_dump()
            self.model = create_siglip_lstm_model(model_architecture_config)
            state_dict = torch.load(self.config.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(self.config.processor_path)
            logger.info(f"✅ Loaded Model: '{self.config.class_name}' | Device: '{self.device}' | Time: {time.time() - start_time:.2f}s.")
        except Exception as e:
            logger.error(f"Failed to load model '{self.config.class_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model '{self.config.class_name}'") from e

class SiglipLSTMV4(BaseSiglipLSTMDetector):
    config: SiglipLSTMv4Config

    def load(self) -> None:
        start_time = time.time()
        try:
            model_architecture_config = self.config.model_definition.model_dump()
            self.model = create_siglip_lstm_model(model_architecture_config)
            state_dict = torch.load(self.config.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(self.config.processor_path)
            logger.info(f"✅ Loaded Model: '{self.config.class_name}' | Device: '{self.device}' | Time: {time.time() - start_time:.2f}s.")
        except Exception as e:
            logger.error(f"Failed to load model '{self.config.class_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model '{self.config.class_name}'") from e