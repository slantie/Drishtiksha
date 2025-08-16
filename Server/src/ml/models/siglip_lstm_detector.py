# src/ml/models/siglip_lstm_detector.py

import os
import time
import torch
import logging
import numpy as np
from PIL import Image
from collections import deque
from typing import Any, Dict, Tuple, Optional
from transformers import AutoProcessor
from tqdm import tqdm
from src.ml.base import BaseModel
from src.ml.utils import extract_frames
from src.ml.event_publisher import publish_progress
from src.config import SiglipLSTMv1Config, SiglipLSTMv3Config
from src.ml.architectures.siglip_lstm import create_lstm_model

logger = logging.getLogger(__name__)
class SiglipLSTMV1(BaseModel):
    config: SiglipLSTMv1Config

    def load(self) -> None:
        start_time = time.time()
        try:
            model_architecture_config = self.config.model_definition.model_dump()
            self.model = create_lstm_model(model_architecture_config)
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
    config: SiglipLSTMv3Config

    def __init__(self, config: SiglipLSTMv3Config):
        super().__init__(config)
        self._last_detailed_result: Optional[Dict[str, Any]] = None
        self._last_video_path: Optional[str] = None

    def _analyze_video_frames(
        self, video_path: str, video_id: str = None, user_id: str = None
    ) -> Tuple[list[float], list[float]]:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file for processing: {video_path}")

        frame_scores = []
        rolling_avg_scores = []
        rolling_window = deque(maxlen=self.config.rolling_window_size)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        try:
            for i in tqdm(range(total_frames), desc=f"Analyzing frames for {self.config.class_name}"):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                inputs = self.processor(images=[frame_pil], return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(self.device)
                with torch.no_grad():
                    logits = self.model(pixel_values, num_frames_per_video=1)
                    prob_fake = torch.sigmoid(logits.squeeze()).item()
                frame_scores.append(prob_fake)
                rolling_window.append(prob_fake)
                rolling_avg_scores.append(np.mean(list(rolling_window)))

                if (i + 1) % 10 == 0 and video_id and user_id:
                    publish_progress(
                        {
                            "videoId": video_id,
                            "userId": user_id,
                            "event": "FRAME_ANALYSIS_PROGRESS",
                            "message": f"Processed frame {i + 1}/{total_frames}",
                            "data": {
                                "modelName": self.config.class_name,
                                "progress": i + 1,
                                "total": total_frames,
                            },
                        }
                    )
        finally:
            cap.release()

        return frame_scores, rolling_avg_scores

    def _get_or_run_detailed_analysis(self, video_path: str, **kwargs) -> Dict[str, Any]:
        if self._last_video_path == video_path and self._last_detailed_result:
            logger.info(f"✅ Using cached detailed analysis for {os.path.basename(video_path)}")
            return self._last_detailed_result

        video_id = kwargs.get("video_id")
        user_id = kwargs.get("user_id")
        start_time = time.time()

        frame_scores, rolling_avg_scores = self._analyze_video_frames(
            video_path, video_id=video_id, user_id=user_id
        )

        if not frame_scores:
            raise ValueError("Frame analysis returned no scores.")

        avg_score = np.mean(frame_scores)
        prediction = "FAKE" if avg_score > 0.5 else "REAL"
        confidence = avg_score if prediction == "FAKE" else 1 - avg_score

        if video_id and user_id:
            publish_progress(
                {
                    "videoId": video_id,
                    "userId": user_id,
                    "event": "FRAME_ANALYSIS_PROGRESS",
                    "message": f"Completed frame analysis for {self.config.class_name}",
                    "data": {
                        "modelName": self.config.class_name,
                        "progress": len(frame_scores),
                        "total": len(frame_scores),
                    },
                }
            )

        result = {
            "prediction": prediction,
            "confidence": confidence,
            "processing_time": time.time() - start_time,
            "metrics": {
                "frame_count": len(frame_scores),
                "per_frame_scores": frame_scores,
                "rolling_average_scores": rolling_avg_scores,
                "final_average_score": avg_score,
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
            {
                "frame_index": i,
                "score": score,
                "prediction": "FAKE" if score > 0.5 else "REAL",
            }
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
        import matplotlib.pyplot as plt

        detailed_result = self._get_or_run_detailed_analysis(video_path, **kwargs)
        metrics = detailed_result["metrics"]
        frame_scores = metrics["per_frame_scores"]

        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        output_temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        output_path = output_temp_file.name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(6, 2.5))

        try:
            for i in range(len(frame_scores)):
                ret, frame = cap.read()
                if not ret:
                    break

                ax.clear()
                ax.fill_between(range(i + 1), frame_scores[: i + 1], color="#FF4136", alpha=0.4)
                ax.plot(range(i + 1), frame_scores[: i + 1], color="#FF851B", linewidth=2)
                ax.axhline(y=0.5, color="white", linestyle="--", alpha=0.7)
                ax.set_ylim(-0.05, 1.05)
                ax.set_xlim(0, len(frame_scores))
                ax.set_title("Suspicion Analysis", fontsize=10)
                fig.tight_layout(pad=1.5)

                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                plot_img_rgba = np.frombuffer(buf, dtype=np.uint8).reshape(
                    fig.canvas.get_width_height()[::-1] + (4,)
                )
                plot_img_bgr = cv2.cvtColor(plot_img_rgba, cv2.COLOR_RGBA2BGR)

                plot_h, plot_w, _ = plot_img_bgr.shape
                new_plot_h = int(frame_height * 0.3)
                new_plot_w = int(new_plot_h * (plot_w / plot_h))
                resized_plot = cv2.resize(plot_img_bgr, (new_plot_w, new_plot_h))

                y_offset, x_offset = 10, frame_width - new_plot_w - 10
                frame[
                    y_offset : y_offset + new_plot_h, x_offset : x_offset + new_plot_w
                ] = resized_plot
                out.write(frame)
        finally:
            cap.release()
            out.release()
            plt.close(fig)

        # Clear the cache after visualization.
        self._last_detailed_result = None
        self._last_video_path = None

        return output_path