# src/ml/models/eyeblink_detector.py

import os
import cv2
import dlib
import time
import torch
import logging
import imutils
import numpy as np
from PIL import Image
from tqdm import tqdm
import tempfile
from imutils import face_utils
from scipy.spatial import distance as dist
from typing import Any, Dict, List, Optional, Tuple
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# REFACTOR: Import the base class and NEW unified schemas.
from src.ml.base import BaseModel, AnalysisResult
from src.app.schemas import VideoAnalysisResult, FramePrediction
from src.config import EyeblinkModelConfig
from src.ml.event_publisher import event_publisher
from src.ml.schemas import ProgressEvent, EventData
from src.ml.architectures.eyeblink_cnn_lstm import create_eyeblink_model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class EyeblinkDetectorV1(BaseModel):
    """
    REFACTORED deepfake detector analyzing eye blink patterns using a CNN+LSTM.
    This class implements the new unified `analyze` method.
    """
    config: EyeblinkModelConfig

    def __init__(self, config: EyeblinkModelConfig):
        super().__init__(config)
        self.shape_predictor: Optional[dlib.shape_predictor] = None
        self.transform: Optional[Compose] = None

    def load(self) -> None:
        """Loads the dlib predictor and the PyTorch CNN+LSTM model."""
        start_time = time.time()
        try:
            if not os.path.exists(str(self.config.dlib_model_path)):
                raise FileNotFoundError(f"Dlib shape predictor not found at: {str(self.config.dlib_model_path)}")
            self.shape_predictor = dlib.shape_predictor(str(self.config.dlib_model_path))

            model_arch_config = self.config.model_definition.model_dump()
            self.model = create_eyeblink_model({**model_arch_config, 'pretrained': True})

            if not os.path.exists(self.config.model_path):
                raise FileNotFoundError(f"PyTorch weights file not found at: {self.config.model_path}")
            state_dict = torch.load(self.config.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=True)
            self.model.to(self.device)
            self.model.eval()

            self.transform = Compose([
                ToTensor(),
                Resize(self.config.model_definition.img_size, antialias=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            load_time = time.time() - start_time
            logger.info(f"Loaded Model: '{self.config.class_name}' | Device: '{self.device}' | Time: {load_time:.2f}s.")
        except Exception as e:
            logger.error(f"Failed to load model '{self.config.class_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model '{self.config.class_name}'") from e

    # --- Private Helper Methods ---

    def _calculate_ear(self, eye: np.ndarray) -> float:
        """Calculates the Eye Aspect Ratio (EAR) for a single eye."""
        y1 = dist.euclidean(eye[1], eye[5])
        y2 = dist.euclidean(eye[2], eye[4])
        x1 = dist.euclidean(eye[0], eye[3])
        return (y1 + y2) / (2.0 * x1) if x1 > 0 else 0.0

    def _extract_blink_frames(self, media_path: str, **kwargs) -> Tuple[List[Image.Image], int]:
        """Scans the video for blinks and extracts relevant frames."""
        video_id = kwargs.get("video_id")
        user_id = kwargs.get("user_id")
        detector = dlib.get_frontal_face_detector()
        (L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        cap = cv2.VideoCapture(media_path)
        if not cap.isOpened(): raise IOError(f"Could not open video file: {media_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = 10  # Analyze every 10th frame for performance
        frames_to_process = range(0, total_frames, frame_skip)

        all_blink_frames: List[Image.Image] = []
        blink_frame_buffer: List[Image.Image] = []
        consecutive_blink_frames = 0

        try:
            for i in tqdm(frames_to_process, desc=f"Detecting blinks for {self.config.class_name}"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret: break

                if video_id and user_id:
                    event_publisher.publish(ProgressEvent(
                        media_id=video_id,
                        user_id=user_id,
                        event="FRAME_ANALYSIS_PROGRESS",
                        message=f"Processed window {i + 1}/{total_frames}",
                        data=EventData(
                            model_name=self.config.class_name,
                            progress=i + 1,
                            total=total_frames
                        )
                    ))

                frame = imutils.resize(frame, width=640)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)

                if faces:
                    shape = self.shape_predictor(gray, faces[0])
                    shape = face_utils.shape_to_np(shape)
                    left_eye, right_eye = shape[L_start:L_end], shape[R_start:R_end]
                    avg_ear = (self._calculate_ear(left_eye) + self._calculate_ear(right_eye)) / 2.0

                    if avg_ear < self.config.blink_threshold:
                        consecutive_blink_frames += 1
                        (x, y, w, h) = cv2.boundingRect(np.concatenate((left_eye, right_eye)))
                        eye_crop = frame[max(0, y - 15):y + h + 15, max(0, x - 15):x + w + 15]
                        if eye_crop.size > 0:
                            blink_frame_buffer.append(Image.fromarray(cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)))
                    else:
                        if consecutive_blink_frames >= self.config.consecutive_frames:
                            all_blink_frames.extend(blink_frame_buffer)
                        consecutive_blink_frames, blink_frame_buffer = 0, []
            
            if consecutive_blink_frames >= self.config.consecutive_frames:
                all_blink_frames.extend(blink_frame_buffer)
        finally:
            cap.release()

        logger.info(f"Detected {len(all_blink_frames)} blink frames in '{os.path.basename(media_path)}'.")
        return all_blink_frames, total_frames

    def _get_sequence_scores(self, media_path: str, **kwargs) -> Tuple[List[float], int, Optional[str]]:
        """Extracts blink frames, runs inference, and returns standardized 'fake' scores."""
        blink_frames, total_frames = self._extract_blink_frames(media_path, **kwargs)

        if len(blink_frames) < self.config.sequence_length:
            note = f"Insufficient blinks detected ({len(blink_frames)} frames found). Result is a low-confidence fallback."
            logger.warning(f"For video '{os.path.basename(media_path)}': {note}")
            return [], total_frames, note

        processed_frames = [self.transform(frame) for frame in blink_frames]
        sequences = [processed_frames[i:i + self.config.sequence_length] for i in range(len(processed_frames) - self.config.sequence_length + 1)]

        if not sequences:
            note = "Could not form analysis sequences from detected blinks."
            return [], total_frames, note

        sequences_tensor = torch.stack([torch.stack(s) for s in sequences]).to(self.device)

        with torch.no_grad():
            logits = self.model(sequences_tensor)
            probs_real = torch.sigmoid(logits.squeeze()).cpu().numpy()
            probs_real = [probs_real.item()] if probs_real.ndim == 0 else probs_real.tolist()

        # STANDARDIZE: Convert "probability of REAL" to "probability of FAKE"
        sequence_scores_fake = [1.0 - p for p in probs_real]
        return sequence_scores_fake, total_frames, None

    def _generate_visualization(self, media_path: str, sequence_scores: List[float], total_frames: int) -> str:
        """Generates a video with an overlay graph of the blink sequence scores."""
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

                seq_len = len(sequence_scores)
                window_index = min(seq_len - 1, int((i / total_frames) * seq_len)) if seq_len > 0 else 0
                score = sequence_scores[window_index] if seq_len > 0 else 0.5

                ax.clear()
                ax.plot(range(seq_len), sequence_scores, color="#FF851B", linewidth=2)
                if seq_len > 0:
                    ax.axvline(x=window_index, color='yellow', linestyle='--', linewidth=2)
                ax.axhline(y=0.5, color="white", linestyle="--", alpha=0.7)
                ax.set_ylim(-0.05, 1.05); ax.set_xlim(0, max(1, seq_len-1))
                ax.set_title("Blink Sequence Suspicion (1.0=Fake)", fontsize=10); fig.tight_layout(pad=1.5)
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

                y_off, x_off = 10, frame_width - new_plot_w - 10
                
                # FIX: Add boundary checks before attempting to overlay the plot.
                if (y_off + new_plot_h <= frame_height and 
                    x_off + new_plot_w <= frame_width and 
                    new_plot_h > 0 and new_plot_w > 0):
                    
                    resized_plot = cv2.resize(plot_img_bgr, (new_plot_w, new_plot_h))
                    frame[y_off:y_off + new_plot_h, x_off:x_off + new_plot_w] = resized_plot
                
                score_color = np.array([0, 0, 255]) * score + np.array([0, 255, 0]) * (1 - score)
                color = tuple(map(int, score_color))
                
                bar_h = 40
                cv2.rectangle(frame, (0, frame_height - bar_h), (frame_width, frame_height), color, -1)
                cv2.putText(frame, f"Blink Sequence Suspicion: {score:.2f}", (15, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                out.write(frame)
        finally:
            cap.release(), out.release(), plt.close(fig)
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
                frames_analyzed=0
            )

        # 3. Calculate final prediction and confidence
        avg_score = np.mean(sequence_scores)
        prediction = "FAKE" if avg_score >= 0.5 else "REAL"
        confidence = avg_score if prediction == "FAKE" else 1 - avg_score

        # 4. Format frame-by-frame predictions (here, frames are blink sequences)
        frame_predictions = [
            FramePrediction(
                index=i,
                score=score,
                prediction="FAKE" if score >= 0.5 else "REAL"
            ) for i, score in enumerate(sequence_scores)
        ]

        # 5. Calculate metrics
        metrics = {
            "blink_sequences_found": len(sequence_scores),
            "suspicious_blink_sequences": sum(1 for s in sequence_scores if s >= 0.5),
            "average_blink_suspicion": avg_score
        }

        # 6. Generate visualization (only if explicitly requested)
        visualization_path = None
        if generate_visualizations:
            visualization_path = self._generate_visualization(media_path, sequence_scores, total_frames)
        else:
            logger.info(f"[{self.config.class_name}] Skipping visualization generation (generate_visualizations=False)")

        # 7. Assemble and return the final result
        return VideoAnalysisResult(
            prediction=prediction,
            confidence=float(confidence),
            processing_time=time.time() - start_time,
            note=note,
            frame_count=total_frames,
            frames_analyzed=len(sequence_scores),
            frame_predictions=frame_predictions,
            metrics=metrics,
            visualization_path=visualization_path
        )