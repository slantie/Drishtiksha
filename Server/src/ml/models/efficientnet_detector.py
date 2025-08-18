# src/ml/models/efficientnet_detector.py

import os
import cv2
import time
import torch
import logging
import warnings
import matplotlib
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
from torchvision.transforms import Normalize
from facenet_pytorch.models.mtcnn import MTCNN
from typing import Any, Dict, List, Optional, Tuple

from src.ml.base import BaseModel
from src.config import EfficientNetB7Config
from src.ml.architectures.efficientnet import create_efficientnet_model
# --- NEW: Import the event publisher ---
from src.ml.event_publisher import publish_progress
# --- END NEW ---

# Update Matplotlib backend
matplotlib.use("Agg")

# Initialize logger
logger = logging.getLogger(__name__)

# Pre-define normalization for image tensors
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean, std)

class EfficientNetB7Detector(BaseModel):
    """
    DeepFake detector using an MTCNN for face extraction and an EfficientNet-B7
    classifier for frame-by-frame, face-by-face analysis.
    """
    config: EfficientNetB7Config

    def __init__(self, config: EfficientNetB7Config):
        super().__init__(config)
        self.face_detector: Optional[MTCNN] = None
        self._last_detailed_result: Optional[Dict[str, Any]] = None
        self._last_video_path: Optional[str] = None

    def load(self) -> None:
        """Loads the EfficientNet model and the MTCNN face detector."""
        start_time = time.time()
        timm_logger = logging.getLogger('timm')
        original_level = timm_logger.level
        try:
            self.face_detector = MTCNN(
                margin=0,
                thresholds=[0.7, 0.8, 0.8],
                device=self.device
            )

            timm_logger.setLevel(logging.WARNING)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.model = create_efficientnet_model(encoder=self.config.encoder)

            timm_logger.setLevel(original_level)

            checkpoint = torch.load(self.config.model_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=True)
            self.model = self.model.to(self.device)
            self.model.eval()

            load_time = time.time() - start_time
            logger.info(
                f"✅ Loaded Model: '{self.config.class_name}'\t | Device: '{self.device}'\t | Time: {load_time:.2f}s."
            )
        except Exception as e:
            # Ensure logger level is restored on failure
            timm_logger.setLevel(original_level)
            logger.error(f"Failed to load model '{self.config.class_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model '{self.config.class_name}'") from e

    def _extract_faces(self, frame: np.ndarray) -> List[np.ndarray]:
        """Detects and extracts all faces from a single frame."""
        faces = []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        batch_boxes, _ = self.face_detector.detect(img, landmarks=False)

        if batch_boxes is None:
            return faces

        for bbox in batch_boxes:
            xmin, ymin, xmax, ymax = [int(b) for b in bbox]
            w, h = xmax - xmin, ymax - ymin
            p_h, p_w = h // 3, w // 3
            crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
            faces.append(crop)
        return faces
        
    def _predict_on_faces(self, faces: List[np.ndarray]) -> List[float]:
        """Runs the classifier on a list of face crops."""
        predictions = []
        if not faces:
            return predictions

        for face in faces:
            if face.size == 0: continue
            
            processed_face = self._preprocess_face(face, self.config.input_size)
            x = torch.tensor(processed_face, device=self.device).float()
            x = x.permute((2, 0, 1))
            x = normalize_transform(x / 255.0)
            x = x.unsqueeze(0)
            
            with torch.no_grad():
                logits = self.model(x)
                prob_fake = torch.sigmoid(logits.squeeze()).item()
                predictions.append(prob_fake)
        return predictions

    def _preprocess_face(self, img: np.ndarray, size: int) -> np.ndarray:
        """Isotropically resizes and pads a face crop."""
        h, w = img.shape[:2]
        scale = size / max(w, h)
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=interp)
        h, w = img.shape[:2]
        padded_img = np.zeros((size, size, 3), dtype=np.uint8)
        start_w, start_h = (size - w) // 2, (size - h) // 2
        padded_img[start_h:start_h + h, start_w:start_w + w] = img
        return padded_img

    def _confident_strategy(self, preds: List[float], threshold: float = 0.8) -> float:
        """Aggregates per-face predictions using a confidence-based heuristic."""
        if not preds: return 0.5
        preds = np.array(preds)
        fakes = np.count_nonzero(preds > threshold)
        if fakes > len(preds) / 2.5 and fakes > 11:
            return np.mean(preds[preds > threshold])
        elif np.count_nonzero(preds < 0.2) > 0.9 * len(preds):
            return np.mean(preds[preds < 0.2])
        else:
            return np.mean(preds)
            
    def predict(self, video_path: str, **kwargs) -> Dict[str, Any]:
        result = self.predict_detailed(video_path, **kwargs)
        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "processing_time": result["processing_time"],
        }

    def _get_or_run_detailed_analysis(self, video_path: str, **kwargs) -> Dict[str, Any]:
        if self._last_video_path == video_path and self._last_detailed_result:
            logger.info(f"✅ Using cached detailed analysis for {os.path.basename(video_path)}")
            return self._last_detailed_result

        # --- NEW: Extract context for progress reporting ---
        video_id = kwargs.get("video_id")
        user_id = kwargs.get("user_id")
        # --- END NEW ---

        start_time = time.time()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise IOError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        all_face_predictions = []
        per_frame_scores = []

        try:
            for i in tqdm(range(total_frames), desc=f"Analyzing faces for {self.config.class_name}"):
                ret, frame = cap.read()
                if not ret: break
                
                faces = self._extract_faces(frame)
                
                if not faces:
                    per_frame_scores.append(0.0)
                    continue

                face_preds = self._predict_on_faces(faces)
                all_face_predictions.extend(face_preds)
                per_frame_scores.append(max(face_preds) if face_preds else 0.0)

                # --- NEW: Emit progress to Redis ---
                if (i + 1) % 10 == 0 and video_id and user_id:
                    publish_progress({
                        "videoId": video_id,
                        "userId": user_id,
                        "event": "FRAME_ANALYSIS_PROGRESS",
                        "message": f"Processed frame {i + 1}/{total_frames}",
                        "data": {
                            "modelName": self.config.class_name,
                            "progress": i + 1,
                            "total": total_frames,
                        },
                    })
                # --- END NEW ---
        finally:
            cap.release()

        # --- NEW: Emit final progress event ---
        if video_id and user_id:
            publish_progress({
                "videoId": video_id,
                "userId": user_id,
                "event": "FRAME_ANALYSIS_PROGRESS",
                "message": f"Completed frame analysis for {self.config.class_name}",
                "data": {
                    "modelName": self.config.class_name,
                    "progress": total_frames,
                    "total": total_frames,
                },
            })
        # --- END NEW ---

        final_prob_fake = self._confident_strategy(all_face_predictions)
        prediction = "FAKE" if final_prob_fake > 0.5 else "REAL"
        confidence = final_prob_fake if prediction == "FAKE" else 1 - final_prob_fake
        processing_time = time.time() - start_time

        result = {
            "prediction": prediction,
            "confidence": confidence,
            "processing_time": processing_time,
            "metrics": {
                "frame_count": total_frames,
                "total_faces_detected": len(all_face_predictions),
                "per_frame_scores": per_frame_scores,
                "average_face_score": np.mean(all_face_predictions) if all_face_predictions else 0.0,
                "max_score": max(per_frame_scores) if per_frame_scores else 0.0,
                "min_score": min(per_frame_scores) if per_frame_scores else 0.0,
                "suspicious_frames_count": sum(1 for s in per_frame_scores if s > 0.5),
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
            "temporal_analysis": {}, # EfficientNet is frame-based, no temporal data
        }

    def predict_visual(self, video_path: str, **kwargs) -> str:
        detailed_result = self._get_or_run_detailed_analysis(video_path, **kwargs)
        frame_scores = detailed_result["metrics"]["per_frame_scores"]

        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        import tempfile
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
            out.release()
            plt.close(fig)

        self._last_detailed_result = None
        self._last_video_path = None
        return output_path