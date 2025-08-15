# src/ml/models/siglip_lstm_detector.py

import time
import torch
import logging
import numpy as np
from PIL import Image
from collections import deque
from typing import Any, Dict, Tuple
from transformers import AutoProcessor

from src.ml.base import BaseModel
from src.ml.utils import extract_frames
from src.config import SiglipLSTMv1Config, SiglipLSTMv3Config
from src.ml.architectures.siglip_lstm import create_lstm_model

logger = logging.getLogger(__name__)


class SiglipLSTMV1(BaseModel):
    """
    Implementation of the SigLIP-LSTM v1 detector.
    Provides the core `predict` capability.
    """
    config: SiglipLSTMv1Config

    def load(self) -> None:
        """Loads the LSTM model, weights, and processor."""
        start_time = time.time()
        logger.info(f"Loading model '{self.config.class_name}' on device '{self.device}'...")

        try:
            model_architecture_config = self.config.model_definition.model_dump()
            
            self.model = create_lstm_model(model_architecture_config)
            
            state_dict = torch.load(self.config.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            self.processor = AutoProcessor.from_pretrained(self.config.processor_path)
            
            load_time = time.time() - start_time
            logger.info(
                f"✅ Model '{self.config.class_name}' loaded successfully "
                f"in {load_time:.2f} seconds."
            )
        except Exception as e:
            logger.error(f"Failed to load model '{self.config.class_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model '{self.config.class_name}'") from e

    def predict(self, video_path: str) -> Dict[str, Any]:
        """Analyzes a video using a batch of frames for a quick prediction."""
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
    Enhanced V3 detector inheriting from V1.
    Adds detailed, frame-by-frame, and visual analysis capabilities.
    """
    config: SiglipLSTMv3Config

    def _analyze_video_frames(self, video_path: str) -> Tuple[list[float], list[float]]:
        """
        Helper method to analyze a video frame-by-frame.

        Returns:
            A tuple containing (list of per-frame scores, list of rolling average scores).
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file for processing: {video_path}")
        
        frame_scores = []
        rolling_avg_scores = []
        rolling_window = deque(maxlen=self.config.rolling_window_size)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Analyzing {total_frames} frames for '{self.config.class_name}'...")

        try:
            for _ in range(total_frames):
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
        finally:
            cap.release()
            
        return frame_scores, rolling_avg_scores

    def predict_detailed(self, video_path: str) -> Dict[str, Any]:
        """Provides a detailed analysis with comprehensive metrics."""
        start_time = time.time()
        frame_scores, rolling_avg_scores = self._analyze_video_frames(video_path)

        if not frame_scores:
            raise ValueError("Frame analysis returned no scores.")

        avg_score = np.mean(frame_scores)
        prediction = "FAKE" if avg_score > 0.5 else "REAL"
        confidence = avg_score if prediction == "FAKE" else 1 - avg_score
        
        return {
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

    def predict_frames(self, video_path: str) -> Dict[str, Any]:
        """Provides per-frame predictions and temporal analysis."""
        detailed_result = self.predict_detailed(video_path)
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
            }
        }

    def predict_visual(self, video_path: str) -> str:
        """Generates a video with an overlaid analysis graph."""
        import cv2
        import tempfile
        import matplotlib.pyplot as plt

        metrics = self.predict_detailed(video_path)["metrics"]
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
                ax.fill_between(range(i + 1), frame_scores[:i+1], color="#FF4136", alpha=0.4)
                ax.plot(range(i + 1), frame_scores[:i+1], color="#FF851B", linewidth=2)
                ax.axhline(y=0.5, color="white", linestyle="--", alpha=0.7)
                ax.set_ylim(-0.05, 1.05)
                ax.set_xlim(0, len(frame_scores))
                ax.set_title("Suspicion Analysis", fontsize=10)
                fig.tight_layout(pad=1.5)

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
                out.write(frame)
        finally:
            cap.release()
            out.release()
            plt.close(fig)

        return output_path