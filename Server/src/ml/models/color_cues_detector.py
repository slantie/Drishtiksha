# src/ml/models/color_cues_detector.py

import matplotlib
matplotlib.use("Agg")

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

from src.ml.base import BaseModel
from src.config import ColorCuesConfig
from src.ml.event_publisher import publish_progress
from src.ml.architectures.color_cues_lstm import create_color_cues_model

logger = logging.getLogger(__name__)


class ColorCuesLSTMV1(BaseModel):
    """Color Cues + LSTM video deepfake detector.

    Pipeline:
      1) Sample frames uniformly across the video.
      2) Detect face with dlib; crop around facial landmarks with a safety margin.
      3) Compute normalized (r, g) chromaticity histogram features per frame.
      4) Build overlapping sequences of histograms and score with the LSTM model.
    """

    config: ColorCuesConfig

    def __init__(self, config: ColorCuesConfig):
        super().__init__(config)
        self.dlib_detector: Optional[Any] = None
        self.dlib_predictor: Optional[Any] = None
        # Cached detailed result to avoid recomputation across helpers
        self._last_detailed_result: Optional[Dict[str, Any]] = None
        self._last_video_path: Optional[str] = None

    # -------------------------
    # Loading
    # -------------------------
    def load(self) -> None:
        start_time = time.time()
        try:
            if not os.path.exists(self.config.dlib_model_path):
                raise FileNotFoundError(
                    f"Dlib model not found at: {self.config.dlib_model_path}"
                )

            # dlib face detector + landmark predictor
            self.dlib_detector = dlib.get_frontal_face_detector()
            self.dlib_predictor = dlib.shape_predictor(self.config.dlib_model_path)

            # Torch model
            model_params = self.config.model_dump()
            self.model = create_color_cues_model(model_params).to(self.device)

            checkpoint = torch.load(self.config.model_path, map_location=self.device)
            state_dict_to_load = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict_to_load)
            self.model.eval()

            load_time = time.time() - start_time
            logger.info(
                "✅ Loaded Model: '%s'\t | Device: '%s'\t | Time: %.2f seconds.",
                self.config.class_name,
                self.device,
                load_time,
            )
        except Exception as e:
            logger.error(
                "Failed to load model '%s': %s",
                self.config.class_name,
                e,
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to load model '{self.config.class_name}'"
            ) from e

    # -------------------------
    # Feature Extraction
    # -------------------------
    def _extract_features_from_video(
        self, video_path: str, video_id: str = None, user_id: str = None
    ) -> List[np.ndarray]:
        # --- NEW: RETRY MECHANISM ---
        max_retries = 3
        retry_delay = 0.5  # seconds
        cap = None

        for attempt in range(max_retries):
            try:
                # Add a small delay to allow the OS to flush file buffers
                time.sleep(attempt * retry_delay) 
                
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    logger.info(f"Successfully opened video file on attempt {attempt + 1}")
                    break # Exit loop on success
                else:
                    logger.warning(f"Attempt {attempt + 1}: cv2.VideoCapture failed to open file. Retrying...")
            except Exception as e:
                 logger.error(f"Attempt {attempt + 1}: Exception during VideoCapture: {e}")
            
            if attempt == max_retries - 1:
                # If all retries fail, raise the final error
                raise IOError(f"Could not open video file after {max_retries} attempts: {video_path}")
        # --- END NEW: RETRY MECHANISM ---
        
        all_histograms: List[np.ndarray] = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return all_histograms

        # ... (rest of the function remains the same) ...
        frames_to_take = max(1, int(self.config.frames_per_video))
        frame_indices = np.linspace(0, total_frames - 1, frames_to_take, dtype=int)

        try:
            for i, frame_idx in enumerate(
                tqdm(frame_indices, desc=f"Analyzing frames for {self.config.class_name}")
            ):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ret, frame = cap.read()
                if not ret:
                    continue

                # Progress events (lightweight cadence)
                if (i + 1) % 5 == 0 and video_id and user_id:
                    publish_progress(
                        {
                            "videoId": video_id,
                            "userId": user_id,
                            "event": "FRAME_ANALYSIS_PROGRESS",
                            "message": f"Processed frame {i + 1}/{len(frame_indices)}",
                            "data": {
                                "modelName": self.config.class_name,
                                "progress": i + 1,
                                "total": len(frame_indices),
                            },
                        }
                    )

                # Face detection & landmarks
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.dlib_detector(gray_frame, 1)
                if not faces:
                    continue

                shape = self.dlib_predictor(gray_frame, faces[0])
                landmarks = np.array([(p.x, p.y) for p in shape.parts()])
                margin = int(self.config.landmark_margin)

                min_coords = np.min(landmarks, axis=0)
                max_coords = np.max(landmarks, axis=0)
                x1 = max(0, int(min_coords[0] - margin))
                y1 = max(0, int(min_coords[1] - margin))
                x2 = min(frame.shape[1], int(max_coords[0] + margin))
                y2 = min(frame.shape[0], int(max_coords[1] + margin))

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                # Chromaticity (r, g) with numeric stability
                img_float = face_crop.astype(np.float32) + 1e-6
                rgb_sum = np.sum(img_float, axis=2)
                rgb_sum[rgb_sum == 0] = 1
                r = img_float[:, :, 2] / rgb_sum
                g = img_float[:, :, 1] / rgb_sum

                # 2D histogram in (r, g)
                hist, _, _ = np.histogram2d(
                    r.flatten(),
                    g.flatten(),
                    bins=int(self.config.histogram_bins),
                    range=[[0, 1], [0, 1]],
                )
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                all_histograms.append(hist.astype(np.float32))
        finally:
            cap.release()

        return all_histograms

    # -------------------------
    # Sequence Analysis
    # -------------------------
    def _analyze_sequences(
        self, video_path: str, video_id: str = None, user_id: str = None
    ) -> Tuple[List[float], List[float], Optional[str]]:
        all_histograms = self._extract_features_from_video(video_path, video_id, user_id)

        if len(all_histograms) < int(self.config.sequence_length):
            logger.warning(
                "Could not extract sufficient features from '%s'. Required %d, found %d. Returning a default low score.",
                os.path.basename(video_path),
                int(self.config.sequence_length),
                len(all_histograms),
            )
            note = (
                "Could not find a clear face for a sufficient duration; result is a low-confidence fallback."
            )
            return [0.1], [0.1], note

        # Build overlapping sequences
        seq_len = int(self.config.sequence_length)
        sub_sequences = [
            all_histograms[i : i + seq_len]
            for i in range(len(all_histograms) - seq_len + 1)
        ]

        if not sub_sequences:
            note = "Could not create analysis sequences from extracted features."
            return [0.1], [0.1], note

        seq_array = np.array(sub_sequences, dtype=np.float32)
        sequences_tensor = torch.from_numpy(seq_array).to(self.device)

        with torch.no_grad():
            logits = self.model(sequences_tensor).squeeze()
            predictions = torch.sigmoid(logits).detach().cpu().numpy()

        if isinstance(predictions, np.ndarray):
            per_sequence_scores: List[float] = predictions.astype(float).tolist()
        else:
            per_sequence_scores = [float(predictions)]

        # Rolling average with fixed window
        rolling_averages: List[float] = []
        rolling_window: deque = deque(maxlen=int(self.config.rolling_window_size))
        for score in per_sequence_scores:
            rolling_window.append(float(score))
            rolling_averages.append(float(np.mean(list(rolling_window))))

        return per_sequence_scores, rolling_averages, None

    # -------------------------
    # Gatekeeper (cache-aware)
    # -------------------------
    def _get_or_run_detailed_analysis(self, video_path: str, **kwargs) -> Dict[str, Any]:
        """Run analysis once per video and cache the result until visualization is produced."""
        if self._last_video_path == video_path and self._last_detailed_result:
            logger.info(
                "✅ Using cached detailed analysis for %s", os.path.basename(video_path)
            )
            return self._last_detailed_result

        video_id = kwargs.get("video_id")
        user_id = kwargs.get("user_id")
        start_time = time.time()

        sequence_scores, rolling_avg_scores, note = self._analyze_sequences(
            video_path, video_id=video_id, user_id=user_id
        )

        avg_score = float(np.mean(sequence_scores)) if sequence_scores else 0.0
        prediction = "FAKE" if avg_score > 0.5 else "REAL"
        confidence = avg_score if prediction == "FAKE" else 1 - avg_score

        # Final progress event for this stage
        if video_id and user_id:
            publish_progress(
                {
                    "videoId": video_id,
                    "userId": user_id,
                    "event": "FRAME_ANALYSIS_PROGRESS",
                    "message": f"Completed frame analysis for {self.config.class_name}",
                    "data": {
                        "modelName": self.config.class_name,
                        "progress": len(sequence_scores),
                        "total": len(sequence_scores),
                    },
                }
            )

        result: Dict[str, Any] = {
            "prediction": prediction,
            "confidence": float(confidence),
            "processing_time": float(time.time() - start_time),
            "metrics": {
                "sequence_count": len(sequence_scores),
                # For downstream consumers expecting frame-like keys
                "frame_count": len(sequence_scores),
                "per_sequence_scores": sequence_scores,
                "per_frame_scores": sequence_scores,
                "rolling_average_scores": rolling_avg_scores,
                "final_average_score": avg_score,
                "max_score": max(sequence_scores) if sequence_scores else 0.0,
                "min_score": min(sequence_scores) if sequence_scores else 0.0,
                "score_variance": float(np.var(sequence_scores)) if sequence_scores else 0.0,
                "suspicious_sequences_count": int(
                    sum(1 for s in sequence_scores if s > 0.5)
                ),
                "suspicious_frames_count": int(
                    sum(1 for s in sequence_scores if s > 0.5)
                ),
                "analysis_type": "sequence_based",
            },
            "note": note,
        }

        self._last_video_path = video_path
        self._last_detailed_result = result
        return result

    # -------------------------
    # Public APIs
    # -------------------------
    def predict(self, video_path: str, **kwargs) -> Dict[str, Any]:
        result = self._get_or_run_detailed_analysis(video_path, **kwargs)
        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "processing_time": result["processing_time"],
            "note": result.get("note"),
        }

    def predict_detailed(self, video_path: str, **kwargs) -> Dict[str, Any]:
        """Performs a detailed, cache-aware analysis."""
        return self._get_or_run_detailed_analysis(video_path, **kwargs)

    def predict_frames(self, video_path: str, **kwargs) -> Dict[str, Any]:
        """Returns per-sequence scores, formatted like per-frame predictions."""
        detailed_result = self._get_or_run_detailed_analysis(video_path, **kwargs)
        metrics = detailed_result["metrics"]
        frame_predictions = [
            {
                "frame_index": i,
                "score": float(score),
                "prediction": "FAKE" if score > 0.5 else "REAL",
            }
            for i, score in enumerate(metrics["per_sequence_scores"])
        ]
        return {
            "overall_prediction": detailed_result["prediction"],
            "overall_confidence": detailed_result["confidence"],
            "processing_time": detailed_result["processing_time"],
            "frame_predictions": frame_predictions,
            "temporal_analysis": {
                "rolling_averages": metrics["rolling_average_scores"],
                "consistency_score": 1.0 - metrics["score_variance"],
                "note": "Analysis is based on overlapping sequences of frames.",
            },
            "note": detailed_result.get("note"),
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

        # Clear the cache after visualization to force a fresh pass next time
        self._last_detailed_result = None
        self._last_video_path = None

        return output_path