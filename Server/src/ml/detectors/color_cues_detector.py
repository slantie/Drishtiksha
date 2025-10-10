# src/ml/models/color_cues_detector.py

import os
import cv2
import dlib
import time
import torch
import logging
import tempfile
import numpy as np
from tqdm import tqdm
from collections import deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple

# REFACTOR: Import the base class and NEW unified schemas.
from src.ml.base import BaseModel, AnalysisResult
from src.app.schemas import VideoAnalysisResult, FramePrediction
from src.config import ColorCuesConfig
from src.ml.event_publisher import event_publisher
from src.ml.schemas import ProgressEvent, EventData
from src.ml.architectures.color_cues_lstm import create_color_cues_model

logger = logging.getLogger(__name__)


class ColorCuesLSTMV1(BaseModel):
    """
    REFACTORED Color Cues + LSTM video deepfake detector.
    This class implements the new unified `analyze` method.
    """
    config: ColorCuesConfig

    def __init__(self, config: ColorCuesConfig):
        super().__init__(config)
        self.dlib_detector: Optional[dlib.fhog_object_detector] = None
        self.dlib_predictor: Optional[dlib.shape_predictor] = None

    def load(self) -> None:
        """Loads the dlib landmark predictor and the PyTorch LSTM model."""
        start_time = time.time()
        try:
            if not os.path.exists(str(self.config.dlib_model_path)):
                raise FileNotFoundError(f"Dlib landmark model not found at: {str(self.config.dlib_model_path)}")

            self.dlib_detector = dlib.get_frontal_face_detector()
            self.dlib_predictor = dlib.shape_predictor(str(self.config.dlib_model_path))

            model_params = self.config.model_dump()
            self.model = create_color_cues_model(model_params).to(self.device)

            checkpoint = torch.load(self.config.model_path, map_location=self.device)
            state_dict_to_load = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict_to_load)
            self.model.eval()

            load_time = time.time() - start_time
            logger.info(f"✅ Loaded Model: '{self.config.model_name}' | Device: '{self.device}' | Time: {load_time:.2f}s.")
        except Exception as e:
            logger.error(f"Failed to load model '{self.config.model_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model '{self.config.model_name}'") from e

    # --- Private Helper Methods ---

    def _get_sequence_scores(self, media_path: str, **kwargs) -> Tuple[List[float], int, Optional[str]]:
        """Extracts features, runs inference, and returns raw temporal scores."""
        video_id = kwargs.get("video_id")
        user_id = kwargs.get("user_id")

        cap = cv2.VideoCapture(media_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {media_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return [], 0, "Video file appears to be empty or corrupt."

        all_histograms: List[np.ndarray] = []
        frames_to_sample = max(1, self.config.frames_per_video)
        frame_indices = np.linspace(0, total_frames - 1, frames_to_sample, dtype=int)

        try:
            for i, frame_idx in enumerate(tqdm(frame_indices, desc=f"Analyzing frames for {self.config.model_name}")):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ret, frame = cap.read()
                if not ret: continue

                if (i + 1) % 5 == 0 and video_id and user_id:
                    event_publisher.publish(ProgressEvent(
                        media_id=video_id,
                        user_id=user_id,
                        event="FRAME_ANALYSIS_PROGRESS",
                        message=f"Processed window {i + 1}/{frame_indices}",
                        data=EventData(
                            model_name=self.config.model_name,
                            progress=i + 1,
                            total=frame_indices
                        )
                    ))

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.dlib_detector(gray_frame, 1)
                if not faces: continue

                shape = self.dlib_predictor(gray_frame, faces[0])
                landmarks = np.array([(p.x, p.y) for p in shape.parts()])
                margin = self.config.landmark_margin

                x1, y1 = np.min(landmarks, axis=0) - margin
                x2, y2 = np.max(landmarks, axis=0) + margin
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0: continue

                img_float = face_crop.astype(np.float32) + 1e-6
                rgb_sum = np.sum(img_float, axis=2, keepdims=True)
                rgb_sum[rgb_sum == 0] = 1.0 # Prevent division by zero

                normalized_rgb = img_float / rgb_sum
                r, g = normalized_rgb[:, :, 2].flatten(), normalized_rgb[:, :, 1].flatten()

                hist, _, _ = np.histogram2d(r, g, bins=self.config.histogram_bins, range=[[0, 1], [0, 1]])
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                all_histograms.append(hist.astype(np.float32))
        finally:
            cap.release()

        seq_len = self.config.sequence_length
        if len(all_histograms) < seq_len:
            note = f"Could not extract enough facial features ({len(all_histograms)} found, {seq_len} required). Result is a fallback."
            logger.warning(f"For video '{os.path.basename(media_path)}': {note}")
            return [], total_frames, note

        sub_sequences = [all_histograms[i : i + seq_len] for i in range(len(all_histograms) - seq_len + 1)]
        sequences_tensor = torch.from_numpy(np.array(sub_sequences)).to(self.device)

        with torch.no_grad():
            logits = self.model(sequences_tensor).squeeze()
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            sequence_scores = probs.tolist() if isinstance(probs, np.ndarray) else [float(probs)]

        return sequence_scores, total_frames, None

    def _generate_visualization(
        self,
        media_path: str,
        sequence_scores: List[float],
        total_frames: int
    ) -> str:
        """Generates a video with an overlay graph of the sequence scores."""
        cap = cv2.VideoCapture(media_path)
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
            for i in tqdm(range(total_frames), desc="Generating visualization"):
                ret, frame = cap.read()
                if not ret: break

                current_sequence_idx = 0
                if len(sequence_scores) > 1:
                    current_sequence_idx = min(
                        len(sequence_scores) - 1,
                        int((i / max(1, total_frames)) * len(sequence_scores)),
                    )

                ax.clear()
                ax.plot(range(len(sequence_scores)), sequence_scores, color="#4ECDC4", linewidth=2)
                ax.fill_between(range(len(sequence_scores)), sequence_scores, color="#4ECDC4", alpha=0.3)
                if sequence_scores:
                    ax.plot(current_sequence_idx, sequence_scores[current_sequence_idx], "o", color="yellow", markersize=8)
                ax.axhline(y=0.5, color="white", linestyle="--", alpha=0.7)
                ax.set_ylim(-0.05, 1.05)
                ax.set_xlim(0, len(sequence_scores) -1 if len(sequence_scores) > 1 else 1)
                ax.set_title("Color Cues Sequence Analysis", fontsize=10)
                fig.tight_layout(pad=1.5)
                fig.canvas.draw()

                buf = fig.canvas.buffer_rgba()
                plot_img_rgba = np.frombuffer(buf, dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
                plot_img_bgr = cv2.cvtColor(plot_img_rgba, cv2.COLOR_RGBA2BGR)

                plot_h, plot_w, _ = plot_img_bgr.shape
                new_plot_h = int(frame_height * 0.3)
                new_plot_w = int(new_plot_h * (plot_w / plot_h))
                
                # FIX: Add safety check to ensure the resized plot fits within the frame width.
                if new_plot_w > frame_width - 20:
                    new_plot_w = frame_width - 20
                    new_plot_h = int(new_plot_w * (plot_h / plot_w))

                y_offset, x_offset = 10, frame_width - new_plot_w - 10
                
                # FIX: Add boundary checks before attempting to overlay the plot.
                if (y_offset + new_plot_h <= frame_height and 
                    x_offset + new_plot_w <= frame_width and 
                    new_plot_h > 0 and new_plot_w > 0):
                    
                    resized_plot = cv2.resize(plot_img_bgr, (new_plot_w, new_plot_h))
                    frame[y_offset:y_offset + new_plot_h, x_offset:x_offset + new_plot_w] = resized_plot
                
                out.write(frame)
        finally:
            cap.release()
            out.release()
            plt.close(fig)

        return output_path

    # --- Public API Method ---

    def analyze(self, media_path: str, generate_visualizations: bool = False, **kwargs) -> AnalysisResult:
        """
        The single, unified entry point for running a comprehensive analysis.
        
        Args:
            media_path: Path to the media file
            generate_visualizations: If True, generate visualization video. Defaults to False.
            **kwargs: Additional arguments (video_id, user_id, etc.)
        """
        start_time = time.time()

        # 1. Get temporal scores and metadata
        sequence_scores, total_frames, note = self._get_sequence_scores(media_path, **kwargs)

        # 2. Handle fallback case if analysis could not produce scores
        if not sequence_scores:
            return VideoAnalysisResult(
                prediction="REAL",
                confidence=0.51,
                processing_time=time.time() - start_time,
                note=note,
                frame_count=total_frames,
                frames_analyzed=0,
            )

        # 3. Calculate final prediction and confidence
        avg_score = float(np.mean(sequence_scores))
        prediction = "FAKE" if avg_score > 0.5 else "REAL"
        confidence = avg_score if prediction == "FAKE" else 1 - avg_score

        # 4. Format frame-by-frame predictions
        frame_predictions = [
            FramePrediction(
                index=i,
                score=score,
                prediction="FAKE" if score > 0.5 else "REAL"
            ) for i, score in enumerate(sequence_scores)
        ]

        # 5. Calculate rolling averages and other metrics
        rolling_avg_scores = []
        rolling_window = deque(maxlen=self.config.rolling_window_size)
        for score in sequence_scores:
            rolling_window.append(score)
            rolling_avg_scores.append(float(np.mean(list(rolling_window))))

        metrics = {
            "sequence_count": len(sequence_scores),
            "rolling_average_scores": rolling_avg_scores,
            "final_average_score": avg_score,
            "suspicious_sequences_count": int(sum(1 for s in sequence_scores if s > 0.5)),
        }

        # 6. Generate the visualization video (only if explicitly requested)
        visualization_path = None
        if generate_visualizations:
            visualization_path = self._generate_visualization(media_path, sequence_scores, total_frames)
        else:
            logger.info(f"[{self.config.model_name}] Skipping visualization generation (generate_visualizations=False)")

        # 7. Assemble and return the final, comprehensive result object
        return VideoAnalysisResult(
            prediction=prediction,
            confidence=confidence,
            processing_time=time.time() - start_time,
            note=note,
            frame_count=total_frames,
            frames_analyzed=len(sequence_scores),
            frame_predictions=frame_predictions,
            metrics=metrics,
            visualization_path=visualization_path
        )
