# src/ml/models/color_cues_detector.py

import os
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import dlib
import torch
import numpy as np
from tqdm import tqdm
from collections import deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.ml.base import BaseModel
from src.config import ColorCuesConfig
from src.ml.event_publisher import publish_progress
from src.ml.architectures.color_cues_lstm import create_color_cues_model

logger = logging.getLogger(__name__)


class ColorCuesLSTMV1(BaseModel):
    """Color Cues + LSTM video deepfake detector."""

    config: ColorCuesConfig

    def __init__(self, config: ColorCuesConfig):
        super().__init__(config)
        self.dlib_detector: Optional[Any] = None
        self.dlib_predictor: Optional[Any] = None
        # This simple cache improves performance for sequential API calls (e.g., detailed -> visual)
        self._last_detailed_result: Optional[Dict[str, Any]] = None
        self._last_video_path: Optional[str] = None

    def load(self) -> None:
        """Loads the dlib landmark predictor and the PyTorch LSTM model."""
        start_time = time.time()
        try:
            if not os.path.exists(self.config.dlib_model_path):
                raise FileNotFoundError(f"Dlib landmark model not found at: {self.config.dlib_model_path}")

            self.dlib_detector = dlib.get_frontal_face_detector()
            self.dlib_predictor = dlib.shape_predictor(self.config.dlib_model_path)

            model_params = self.config.model_dump()
            self.model = create_color_cues_model(model_params).to(self.device)

            checkpoint = torch.load(self.config.model_path, map_location=self.device)
            state_dict_to_load = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict_to_load)
            self.model.eval()

            load_time = time.time() - start_time
            logger.info(f"✅ Loaded Model: '{self.config.class_name}' | Device: '{self.device}' | Time: {load_time:.2f}s.")
        except Exception as e:
            logger.error(f"Failed to load model '{self.config.class_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model '{self.config.class_name}'") from e

    def _extract_features_from_video(
        self, video_path: str, video_id: str = None, user_id: str = None
    ) -> List[np.ndarray]:
        """Extracts chromaticity histograms from faces detected in video frames."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")
        
        all_histograms: List[np.ndarray] = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return all_histograms

        frames_to_sample = max(1, int(self.config.frames_per_video))
        frame_indices = np.linspace(0, total_frames - 1, frames_to_sample, dtype=int)

        try:
            for i, frame_idx in enumerate(tqdm(frame_indices, desc=f"Analyzing frames for {self.config.class_name}")):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ret, frame = cap.read()
                if not ret: continue

                if (i + 1) % 5 == 0 and video_id and user_id:
                    publish_progress({
                        "videoId": video_id, "userId": user_id, "event": "FRAME_ANALYSIS_PROGRESS",
                        "message": f"Processed frame {i + 1}/{len(frame_indices)}",
                        "data": {"modelName": self.config.class_name, "progress": i + 1, "total": len(frame_indices)},
                    })

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.dlib_detector(gray_frame, 1)
                if not faces: continue

                shape = self.dlib_predictor(gray_frame, faces[0])
                landmarks = np.array([(p.x, p.y) for p in shape.parts()])
                margin = int(self.config.landmark_margin)
                
                x1, y1 = np.min(landmarks, axis=0) - margin
                x2, y2 = np.max(landmarks, axis=0) + margin
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0: continue

                img_float = face_crop.astype(np.float32) + 1e-6
                rgb_sum = np.sum(img_float, axis=2, keepdims=True)
                
                # Prevent division by zero
                rgb_sum[rgb_sum == 0] = 1.0

                normalized_rgb = img_float / rgb_sum
                r, g = normalized_rgb[:, :, 2].flatten(), normalized_rgb[:, :, 1].flatten()

                hist, _, _ = np.histogram2d(r, g, bins=int(self.config.histogram_bins), range=[[0, 1], [0, 1]])
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                all_histograms.append(hist.astype(np.float32))
        finally:
            cap.release()

        return all_histograms

    def _analyze_sequences(self, video_path: str, **kwargs) -> Tuple[List[float], List[float], Optional[str]]:
        """Analyzes sequences of histograms and returns raw and smoothed scores."""
        all_histograms = self._extract_features_from_video(video_path, **kwargs)
        seq_len = int(self.config.sequence_length)

        # FIX: Robust fallback if feature extraction fails.
        if len(all_histograms) < seq_len:
            note = f"Could not extract enough facial features ({len(all_histograms)} found, {seq_len} required). Result is a fallback."
            logger.warning(f"For video '{os.path.basename(video_path)}': {note}")
            return [], [], note # Return empty lists, handled by caller

        sub_sequences = [all_histograms[i : i + seq_len] for i in range(len(all_histograms) - seq_len + 1)]
        sequences_tensor = torch.from_numpy(np.array(sub_sequences)).to(self.device)

        with torch.no_grad():
            logits = self.model(sequences_tensor).squeeze()
            # Handle single-sequence case where output is not a list
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            per_sequence_scores = probs.tolist() if isinstance(probs, np.ndarray) else [float(probs)]

        rolling_averages: List[float] = []
        rolling_window = deque(maxlen=int(self.config.rolling_window_size))
        for score in per_sequence_scores:
            rolling_window.append(score)
            rolling_averages.append(float(np.mean(list(rolling_window))))

        return per_sequence_scores, rolling_averages, None

    def _get_or_run_detailed_analysis(self, video_path: str, **kwargs) -> Dict[str, Any]:
        """Runs the full analysis pipeline, using a simple cache for efficiency."""
        if self._last_video_path == video_path and self._last_detailed_result:
            logger.info(f"✅ Using cached detailed analysis for {os.path.basename(video_path)}")
            return self._last_detailed_result

        start_time = time.time()
        sequence_scores, rolling_avg_scores, note = self._analyze_sequences(video_path, **kwargs)
        
        # FIX: Standardized fallback logic
        if not sequence_scores:
            result = {
                "prediction": "REAL", "confidence": 0.51, "processing_time": time.time() - start_time,
                "metrics": {"sequence_count": 0, "per_sequence_scores": [], "final_average_score": 0.0}, "note": note
            }
        else:
            avg_score = float(np.mean(sequence_scores))
            prediction = "FAKE" if avg_score > 0.5 else "REAL"
            confidence = avg_score if prediction == "FAKE" else 1 - avg_score
            result = {
                "prediction": prediction, "confidence": float(confidence), "processing_time": time.time() - start_time,
                "metrics": {
                    "sequence_count": len(sequence_scores),
                    "per_sequence_scores": sequence_scores,
                    "rolling_average_scores": rolling_avg_scores,
                    "final_average_score": avg_score,
                    "suspicious_sequences_count": int(sum(1 for s in sequence_scores if s > 0.5)),
                }, "note": note
            }

        self._last_video_path = video_path
        self._last_detailed_result = result
        return result

    # --- Public API methods ---
    def predict(self, video_path: str, **kwargs) -> Dict[str, Any]:
        result = self._get_or_run_detailed_analysis(video_path, **kwargs)
        return {"prediction": result["prediction"], "confidence": result["confidence"], "processing_time": result["processing_time"], "note": result.get("note")}

    def predict_detailed(self, video_path: str, **kwargs) -> Dict[str, Any]:
        return self._get_or_run_detailed_analysis(video_path, **kwargs)

    def predict_frames(self, video_path: str, **kwargs) -> Dict[str, Any]:
        detailed_result = self._get_or_run_detailed_analysis(video_path, **kwargs)
        metrics = detailed_result["metrics"]
        frame_predictions = [{"frame_index": i, "score": float(score), "prediction": "FAKE" if score > 0.5 else "REAL"} for i, score in enumerate(metrics.get("per_sequence_scores", []))]
        return {
            "overall_prediction": detailed_result["prediction"], "overall_confidence": detailed_result["confidence"],
            "processing_time": detailed_result["processing_time"], "frame_predictions": frame_predictions,
            "temporal_analysis": {"rolling_averages": metrics.get("rolling_average_scores", [])}, "note": detailed_result.get("note"),
        }

    def predict_visual(self, video_path: str, **kwargs) -> str:
        """Generate an overlay video visualizing sequence scores over time.

        Cache is cleared after the visualization is produced to avoid stale reuse
        across different visualization passes of the same video.
        """
        import tempfile
        import matplotlib.pyplot as plt

        detailed_result = self._get_or_run_detailed_analysis(video_path, **kwargs)
        metrics = detailed_result["metrics"]
        sequence_scores = metrics["per_sequence_scores"]

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
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                # Map current frame to a sequence index for highlighting
                current_sequence_idx = 0
                if len(sequence_scores) > 1:
                    current_sequence_idx = min(
                        len(sequence_scores) - 1,
                        int((i / max(1, total_frames)) * len(sequence_scores)),
                    )

                ax.clear()
                ax.plot(range(len(sequence_scores)), sequence_scores, color="#4ECDC4", linewidth=2)
                ax.fill_between(
                    range(len(sequence_scores)), sequence_scores, color="#4ECDC4", alpha=0.3
                )
                ax.plot(
                    current_sequence_idx,
                    sequence_scores[current_sequence_idx],
                    "o",
                    color="yellow",
                    markersize=8,
                )
                ax.axhline(y=0.5, color="white", linestyle="--", alpha=0.7)
                ax.set_ylim(-0.05, 1.05)
                ax.set_xlim(0, len(sequence_scores) if len(sequence_scores) > 1 else 1)
                ax.set_title("Color Cues Sequence Analysis", fontsize=10)
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
                frame[y_offset : y_offset + new_plot_h, x_offset : x_offset + new_plot_w] = resized_plot
                out.write(frame)
        finally:
            cap.release()
            out.release()
            plt.close(fig)

        self._last_detailed_result = None
        self._last_video_path = None

        return output_path