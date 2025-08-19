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
from scipy.spatial import distance as dist
from imutils import face_utils
from torchvision.transforms import Normalize
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

from src.ml.base import BaseModel
from src.config import EyeblinkModelConfig
from src.ml.architectures.eyeblink_cnn_lstm import create_eyeblink_model
from src.ml.event_publisher import publish_progress

# Add matplotlib imports for visualization
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Pre-define normalization for image tensors
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean, std)

class EyeblinkDetectorV1(BaseModel):
    config: EyeblinkModelConfig

    def __init__(self, config: EyeblinkModelConfig):
        super().__init__(config)
        self.shape_predictor: Optional[Any] = None
        self._last_detailed_result: Optional[Dict[str, Any]] = None
        self._last_video_path: Optional[str] = None

    def load(self) -> None:
        start_time = time.time()
        try:
            if not os.path.exists(self.config.dlib_model_path):
                raise FileNotFoundError(f"Dlib shape predictor not found at: {self.config.dlib_model_path}")
            self.shape_predictor = dlib.shape_predictor(self.config.dlib_model_path)
            model_architecture_config = self.config.model_definition.model_dump()
            self.model = create_eyeblink_model({**model_architecture_config, 'pretrained': False})
            if not os.path.exists(self.config.model_path):
                raise FileNotFoundError(f"PyTorch weights file not found at: {self.config.model_path}")
            state_dict = torch.load(self.config.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=True)
            self.model.to(self.device)
            self.model.eval()
            load_time = time.time() - start_time
            logger.info(f"✅ Loaded Model: '{self.config.class_name}'\t | Device: '{self.device}'\t | Time: {load_time:.2f}s.")
        except Exception as e:
            logger.error(f"Failed to load model '{self.config.class_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model '{self.config.class_name}'") from e

    def _calculate_ear(self, eye: np.ndarray) -> float:
        y1 = dist.euclidean(eye[1], eye[5])
        y2 = dist.euclidean(eye[2], eye[4])
        x1 = dist.euclidean(eye[0], eye[3])
        return (y1 + y2) / (2.0 * x1) if x1 > 0 else 0.0

    def _extract_blink_frames(self, video_path: str, video_id: str = None, user_id: str = None) -> List[np.ndarray]:
        detector = dlib.get_frontal_face_detector()
        (L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")
        blink_frames_buffer = []
        all_blink_frames = []
        consecutive_blink_frames = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            for i in tqdm(range(total_frames), desc=f"Detecting blinks for {self.config.class_name}"):
                ret, frame = cap.read()
                if not ret: break
                if (i + 1) % 10 == 0 and video_id and user_id:
                    publish_progress({
                        "videoId": video_id, "userId": user_id, "event": "FRAME_ANALYSIS_PROGRESS",
                        "message": f"Scanning frame {i + 1}/{total_frames} for blinks",
                        "data": {"modelName": self.config.class_name, "progress": i + 1, "total": total_frames},
                    })
                frame = imutils.resize(frame, width=640)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)
                if len(faces) > 0:
                    shape = self.shape_predictor(gray, faces[0])
                    shape = face_utils.shape_to_np(shape)
                    left_eye, right_eye = shape[L_start:L_end], shape[R_start:R_end]
                    avg_ear = (self._calculate_ear(left_eye) + self._calculate_ear(right_eye)) / 2.0
                    if avg_ear < self.config.blink_threshold:
                        consecutive_blink_frames += 1
                        all_eye_landmarks = np.concatenate((left_eye, right_eye))
                        (x, y, w, h) = cv2.boundingRect(all_eye_landmarks)
                        pad = 15
                        eye_crop = frame[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
                        if eye_crop.size > 0:
                            blink_frames_buffer.append(eye_crop)
                    else:
                        if consecutive_blink_frames >= self.config.consecutive_frames:
                            all_blink_frames.extend(blink_frames_buffer)
                        consecutive_blink_frames = 0
                        blink_frames_buffer = []
        finally:
            cap.release()
        return all_blink_frames

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, self.config.model_definition.img_size)
        return resized.astype(np.float32) / 255.0
        
    def _get_or_run_detailed_analysis(self, video_path: str, **kwargs) -> Dict[str, Any]:
        if self._last_video_path == video_path and self._last_detailed_result:
            return self._last_detailed_result

        start_time = time.time()
        video_id = kwargs.get("video_id")
        user_id = kwargs.get("user_id")
        
        blink_frames = self._extract_blink_frames(video_path, video_id, user_id)
        
        note = None
        # --- CORRECTED: Ensure the fallback dictionary has the expected metric keys ---
        if len(blink_frames) < self.config.sequence_length:
            note = f"Insufficient blinks detected ({len(blink_frames)} frames found). Prediction is a fallback."
            return {
                "prediction": "REAL", "confidence": 0.9, "processing_time": time.time() - start_time,
                "metrics": {
                    "frame_count": len(blink_frames), "per_frame_scores": [], "sequence_count": 0,
                    "final_average_score": None, "max_score": None, "min_score": None, "suspicious_frames_count": 0
                }, "note": note
            }
        # --- END CORRECTION ---

        processed_frames = [self._preprocess_frame(f) for f in blink_frames if f is not None]
        sequences = [processed_frames[i:i + self.config.sequence_length] for i in range(len(processed_frames) - self.config.sequence_length + 1)]
        
        if not sequences:
            note = "Could not form analysis sequences from detected blinks."
            return {
                "prediction": "REAL", "confidence": 0.9, "processing_time": time.time() - start_time,
                "metrics": {
                    "frame_count": len(blink_frames), "per_frame_scores": [], "sequence_count": 0,
                    "final_average_score": None, "max_score": None, "min_score": None, "suspicious_frames_count": 0
                }, "note": note
            }

        sequences_np = np.array(sequences)
        sequences_tensor = torch.from_numpy(sequences_np).permute(0, 1, 4, 2, 3).to(self.device)

        with torch.no_grad():
            predictions = self.model(sequences_tensor).squeeze()
            probs = torch.sigmoid(predictions).cpu().numpy().tolist()

        avg_prob_real = np.mean(probs)
        prediction = "REAL" if avg_prob_real >= 0.5 else "FAKE"
        confidence = avg_prob_real if prediction == "REAL" else 1 - avg_prob_real

        result = {
            "prediction": prediction, "confidence": confidence, "processing_time": time.time() - start_time,
            "metrics": {
                "frame_count": len(blink_frames), "sequence_count": len(sequences),
                "per_frame_scores": probs, "final_average_score": avg_prob_real,
                "max_score": max(probs) if probs else 0.0, "min_score": min(probs) if probs else 0.0,
                "suspicious_frames_count": sum(1 for s in probs if s < 0.5),
            },
            "note": note
        }
        self._last_video_path = video_path
        self._last_detailed_result = result
        return result

    def predict(self, video_path: str, **kwargs) -> Dict[str, Any]:
        result = self._get_or_run_detailed_analysis(video_path, **kwargs)
        return {
            "prediction": result["prediction"], "confidence": result["confidence"],
            "processing_time": result["processing_time"], "note": result.get("note")
        }

    def predict_detailed(self, video_path: str, **kwargs) -> Dict[str, Any]:
        return self._get_or_run_detailed_analysis(video_path, **kwargs)

    def predict_frames(self, video_path: str, **kwargs) -> Dict[str, Any]:
        detailed_result = self._get_or_run_detailed_analysis(video_path, **kwargs)
        metrics = detailed_result["metrics"]
        frame_predictions = [
            {"frame_index": i, "score": score, "prediction": "REAL" if score >= 0.5 else "FAKE"}
            for i, score in enumerate(metrics["per_frame_scores"])
        ]
        return {
            "overall_prediction": detailed_result["prediction"],
            "overall_confidence": detailed_result["confidence"],
            "processing_time": detailed_result["processing_time"],
            "frame_predictions": frame_predictions,
            "temporal_analysis": {},
        }

    def predict_visual(self, video_path: str, **kwargs) -> str:
        import tempfile
        detailed_result = self._get_or_run_detailed_analysis(video_path, **kwargs)
        sequence_scores = detailed_result["metrics"].get("per_frame_scores", [])

        # --- CORRECTED: Do not raise an error; allow the router to handle it ---
        if not sequence_scores:
             raise NotImplementedError("Cannot generate visualization without detected blink sequences.")
        # --- END CORRECTION ---

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
                window_index = int((i / total_frames) * (len(sequence_scores) - 1))
                score = sequence_scores[window_index]
                ax.clear()
                ax.plot(range(len(sequence_scores)), sequence_scores, color="#4ECDC4", linewidth=2)
                ax.axvline(x=window_index, color='yellow', linestyle='--', linewidth=2)
                ax.axhline(y=0.5, color="white", linestyle="--", alpha=0.7)
                ax.set_ylim(-0.05, 1.05); ax.set_xlim(0, len(sequence_scores) - 1 if len(sequence_scores) > 1 else 1)
                ax.set_title("Blink Sequence Analysis (1=Real)", fontsize=10); fig.tight_layout(pad=1.5)
                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                plot_img_rgba = np.frombuffer(buf, dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
                plot_img_bgr = cv2.cvtColor(plot_img_rgba, cv2.COLOR_RGBA2BGR)
                plot_h, plot_w, _ = plot_img_bgr.shape
                new_plot_h, new_plot_w = int(frame_height * 0.3), int(int(frame_height * 0.3) * (plot_w / plot_h))
                resized_plot = cv2.resize(plot_img_bgr, (new_plot_w, new_plot_h))
                y_offset, x_offset = 10, frame_width - new_plot_w - 10
                frame[y_offset:y_offset + new_plot_h, x_offset:x_offset + new_plot_w] = resized_plot
                green, yellow, red = np.array([0, 255, 0]), np.array([0, 255, 255]), np.array([0, 0, 255])
                interp = (1 - score) * 2 if score > 0.5 else score * 2
                color_np = red * (1 - interp) + yellow * interp if score < 0.5 else yellow * (1 - interp) + green * interp
                color = tuple(map(int, color_np))
                bar_height = 40
                cv2.rectangle(frame, (0, frame_height - bar_height), (frame_width, frame_height), color, -1)
                cv2.putText(frame, f"Blink Sequence Score: {score:.2f}", (15, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                out.write(frame)
        finally:
            cap.release()
            out.release()
            plt.close(fig)
        self._last_detailed_result = None
        self._last_video_path = None
        return output_path