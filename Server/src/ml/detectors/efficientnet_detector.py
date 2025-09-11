# src/ml/models/efficientnet_detector.py

import os
import cv2
import time
import torch
import logging
import warnings
import tempfile
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.transforms import Normalize
from facenet_pytorch.models.mtcnn import MTCNN
from typing import List, Optional, Tuple

# REFACTOR: Import the base class and NEW unified schemas.
from src.ml.base import BaseModel, AnalysisResult
from src.app.schemas import VideoAnalysisResult, FramePrediction
from src.config import EfficientNetB7Config
from src.ml.architectures.efficientnet import create_efficientnet_model
from src.ml.event_publisher import event_publisher
from src.ml.schemas import ProgressEvent, EventData

import matplotlib
matplotlib.use("Agg")
logger = logging.getLogger(__name__)

# --- Preprocessing setup ---
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean, std)


class EfficientNetB7Detector(BaseModel):
    """
    REFACTORED DeepFake detector using MTCNN for faces and EfficientNet-B7.
    This class implements the new unified `analyze` method.
    """
    config: EfficientNetB7Config

    def __init__(self, config: EfficientNetB7Config):
        super().__init__(config)
        self.face_detector: Optional[MTCNN] = None

    def load(self) -> None:
        """Loads the EfficientNet model and the MTCNN face detector."""
        start_time = time.time()
        try:
            self.face_detector = MTCNN(margin=0, thresholds=[0.7, 0.8, 0.8], device=self.device)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.model = create_efficientnet_model(config=self.config.model_dump())

            checkpoint = torch.load(
                str(self.config.model_path),
                map_location="cpu",
                weights_only=False
            )
            state_dict = checkpoint.get("state_dict", checkpoint)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=True)
            self.model = self.model.to(self.device)
            self.model.eval()

            load_time = time.time() - start_time
            logger.info(f"Loaded Model: '{self.config.class_name}' | Device: '{self.device}' | Time: {load_time:.2f}s.")
        except Exception as e:
            logger.error(f"Failed to load model '{self.config.class_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model '{self.config.class_name}'") from e

    # --- Private Helper Methods ---

    def _extract_faces(self, frame: np.ndarray) -> List[np.ndarray]:
        """Detects and extracts all faces from a single frame."""
        faces = []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        batch_boxes, _ = self.face_detector.detect(img, landmarks=False)
        if batch_boxes is None: return faces

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
        for face in faces:
            if face.size == 0: continue
            processed_face = self._preprocess_face(face, self.config.input_size)
            x = torch.from_numpy(processed_face).to(self.device).float()
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
        if not preds: return 0.0
        preds = np.array(preds)
        fakes = np.count_nonzero(preds > threshold)
        if fakes > len(preds) / 2.5 and fakes > 11:
            return np.mean(preds[preds > threshold])
        if np.count_nonzero(preds < 0.2) > 0.9 * len(preds):
            return np.mean(preds[preds < 0.2])
        return np.mean(preds)

    def _get_frame_scores(self, media_path: str, **kwargs) -> Tuple[List[float], List[float], int, Optional[str]]:
        """Processes the video frame-by-frame, returning raw scores."""
        video_id = kwargs.get("video_id")
        user_id = kwargs.get("user_id")

        cap = cv2.VideoCapture(media_path)
        if not cap.isOpened(): raise IOError(f"Could not open video file: {media_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        all_face_predictions: List[float] = []
        per_frame_scores: List[float] = []

        # Publish initial frame count
        if video_id and user_id:
            event_publisher.publish(ProgressEvent(
                media_id=video_id,
                user_id=user_id,
                event="FRAME_ANALYSIS_PROGRESS",
                message=f"Starting frame analysis for {total_frames} frames",
                data=EventData(
                    model_name=self.config.class_name,
                    progress=0,
                    total=total_frames,
                    details={"phase": "frame_processing_start"}
                )
            ))

        try:
            for i in tqdm(range(total_frames), desc=f"Analyzing frames for {self.config.class_name}"):
                ret, frame = cap.read()
                if not ret: break

                faces = self._extract_faces(frame)
                if not faces:
                    per_frame_scores.append(0.0) # Assume real if no face detected
                    continue

                face_preds = self._predict_on_faces(faces)
                all_face_predictions.extend(face_preds)
                per_frame_scores.append(max(face_preds) if face_preds else 0.0)

                # More frequent progress updates for better real-time feedback
                if (i + 1) % 5 == 0 and video_id and user_id:
                    event_publisher.publish(ProgressEvent(
                        media_id=video_id,
                        user_id=user_id,
                        event="FRAME_ANALYSIS_PROGRESS",
                        message=f"Analyzed {i + 1}/{total_frames} frames, detected {len(all_face_predictions)} faces",
                        data=EventData(
                            model_name=self.config.class_name,
                            progress=i + 1,
                            total=total_frames,
                            details={
                                "phase": "frame_processing",
                                "faces_detected_so_far": len(all_face_predictions),
                                "current_frame_faces": len(faces)
                            }
                        )
                    ))
                    
        finally:
            cap.release()

        note = None
        if not all_face_predictions:
            note = "No faces were detected in the video. Result is a low-confidence fallback."
            logger.warning(f"For video '{os.path.basename(media_path)}': {note}")

        return per_frame_scores, all_face_predictions, total_frames, note

    def _generate_visualization(self, media_path: str, frame_scores: List[float], total_frames: int, **kwargs) -> str:
        """Generates a video with an overlay graph of the frame scores."""
        video_id = kwargs.get("video_id")
        user_id = kwargs.get("user_id")
        
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

        # Publish visualization start event
        if video_id and user_id:
            event_publisher.publish(ProgressEvent(
                media_id=video_id,
                user_id=user_id,
                event="FRAME_ANALYSIS_PROGRESS",
                message="Starting visualization generation",
                data=EventData(
                    model_name=self.config.class_name,
                    progress=0,
                    total=total_frames,
                    details={"phase": "visualization"}
                )
            ))

        try:
            for i in tqdm(range(total_frames), desc="Generating visualization"):
                ret, frame = cap.read()
                if not ret: break
                
                score = frame_scores[i] if i < len(frame_scores) else 0.0

                ax.clear()
                ax.fill_between(range(i + 1), frame_scores[: i + 1], color="#FF4136", alpha=0.4)
                ax.plot(range(i + 1), frame_scores[: i + 1], color="#FF851B", linewidth=2)
                ax.axhline(y=0.5, color="white", linestyle="--", alpha=0.7)
                ax.set_ylim(-0.05, 1.05); ax.set_xlim(0, max(1, len(frame_scores) -1))
                ax.set_title("Frame Suspicion Analysis", fontsize=10); fig.tight_layout(pad=1.5)
                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                plot_img_rgba = np.frombuffer(buf, dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
                plot_img_bgr = cv2.cvtColor(plot_img_rgba, cv2.COLOR_RGBA2BGR)
                
                plot_h, plot_w, _ = plot_img_bgr.shape
                new_plot_h = int(frame_height * 0.3)
                new_plot_w = int(new_plot_h * (plot_w / plot_h))
                
                # FIX: Ensure the plot fits within frame boundaries
                if new_plot_w > frame_width - 20:  # Leave 10px margin on each side
                    new_plot_w = frame_width - 20
                    new_plot_h = int(new_plot_w * (plot_h / plot_w))
                
                # FIX: Ensure coordinates don't exceed frame boundaries
                x_offset = max(0, min(frame_width - new_plot_w - 10, frame_width - new_plot_w - 10))
                y_offset = 10
                
                # FIX: Double-check dimensions before assignment
                if (y_offset + new_plot_h <= frame_height and 
                    x_offset + new_plot_w <= frame_width and 
                    new_plot_h > 0 and new_plot_w > 0):
                    
                    resized_plot = cv2.resize(plot_img_bgr, (new_plot_w, new_plot_h))
                    frame[y_offset:y_offset + new_plot_h, x_offset:x_offset + new_plot_w] = resized_plot

                score_color = np.array([0, 0, 255]) * score + np.array([0, 255, 0]) * (1 - score)
                color = tuple(map(int, score_color))
                
                bar_height = 40
                cv2.rectangle(frame, (0, frame_height - bar_height), (frame_width, frame_height), color, -1)
                cv2.putText(frame, f"Frame Suspicion: {score:.2f}", (15, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                out.write(frame)
                
                # Publish visualization progress updates
                if (i + 1) % 50 == 0 and video_id and user_id:
                    event_publisher.publish(ProgressEvent(
                        media_id=video_id,
                        user_id=user_id,
                        event="FRAME_ANALYSIS_PROGRESS",
                        message=f"Generating visualization: {i + 1}/{total_frames} frames processed",
                        data=EventData(
                            model_name=self.config.class_name,
                            progress=i + 1,
                            total=total_frames,
                            details={"phase": "visualization"}
                        )
                    ))
        finally:
            cap.release()
            out.release()
            plt.close(fig)
            
        # Publish visualization completion event
        if video_id and user_id:
            event_publisher.publish(ProgressEvent(
                media_id=video_id,
                user_id=user_id,
                event="FRAME_ANALYSIS_PROGRESS",
                message="Visualization generation completed",
                data=EventData(
                    model_name=self.config.class_name,
                    progress=total_frames,
                    total=total_frames,
                    details={"phase": "visualization_complete"}
                )
            ))
            
        return output_path

    # --- Public API Method ---

    def analyze(self, media_path: str, **kwargs) -> AnalysisResult:
        """The single, unified entry point for running a comprehensive analysis."""
        start_time = time.time()
        video_id = kwargs.get("video_id")
        user_id = kwargs.get("user_id")

        # Publish analysis start event
        if video_id and user_id:
            event_publisher.publish(ProgressEvent(
                media_id=video_id,
                user_id=user_id,
                event="FRAME_ANALYSIS_PROGRESS",
                message=f"Starting analysis with {self.config.class_name}",
                data=EventData(
                    model_name=self.config.class_name,
                    progress=0,
                    total=None,
                    details={"phase": "initialization"}
                )
            ))

        try:
            # 1. Get temporal scores and metadata
            frame_scores, all_face_scores, total_frames, note = self._get_frame_scores(media_path, **kwargs)
            
            # Publish frame analysis completion
            if video_id and user_id:
                event_publisher.publish(ProgressEvent(
                    media_id=video_id,
                    user_id=user_id,
                    event="FRAME_ANALYSIS_PROGRESS",
                    message=f"Frame analysis completed. Processing {len(all_face_scores)} face detections",
                    data=EventData(
                        model_name=self.config.class_name,
                        progress=total_frames,
                        total=total_frames,
                        details={"phase": "frame_analysis_complete", "faces_detected": len(all_face_scores)}
                    )
                ))
            
            # 2. Calculate final prediction using the aggregation strategy
            final_prob_fake = self._confident_strategy(all_face_scores)
            prediction = "FAKE" if final_prob_fake > 0.5 else "REAL"
            confidence = final_prob_fake if prediction == "FAKE" else 1 - final_prob_fake

            # 3. Format frame-by-frame predictions
            frame_predictions = [
                FramePrediction(
                    index=i,
                    score=score,
                    prediction="FAKE" if score > 0.5 else "REAL"
                ) for i, score in enumerate(frame_scores)
            ]

            # 4. Calculate metrics
            metrics = {
                "total_faces_detected": len(all_face_scores),
                "average_face_score": np.mean(all_face_scores) if all_face_scores else 0.0,
                "suspicious_frames_count": sum(1 for s in frame_scores if s > 0.5),
            }

            # 5. Generate visualization
            visualization_path = self._generate_visualization(media_path, frame_scores, total_frames, **kwargs)

            # 6. Publish analysis completion
            if video_id and user_id:
                event_publisher.publish(ProgressEvent(
                    media_id=video_id,
                    user_id=user_id,
                    event="ANALYSIS_COMPLETE",
                    message=f"Analysis completed: {prediction} (confidence: {confidence:.3f})",
                    data=EventData(
                        model_name=self.config.class_name,
                        progress=total_frames,
                        total=total_frames,
                        details={
                            "prediction": prediction,
                            "confidence": confidence,
                            "processing_time": time.time() - start_time,
                            "total_frames": total_frames,
                            "faces_detected": len(all_face_scores)
                        }
                    )
                ))

            # 7. Assemble and return the final result
            return VideoAnalysisResult(
                prediction=prediction,
                confidence=confidence,
                processing_time=time.time() - start_time,
                note=note,
                frame_count=total_frames,
                frames_analyzed=len(frame_scores),
                frame_predictions=frame_predictions,
                metrics=metrics,
                visualization_path=visualization_path
            )
            
        except Exception as e:
            # Publish analysis failure event
            if video_id and user_id:
                event_publisher.publish(ProgressEvent(
                    media_id=video_id,
                    user_id=user_id,
                    event="ANALYSIS_FAILED",
                    message=f"Analysis failed: {str(e)}",
                    data=EventData(
                        model_name=self.config.class_name,
                        details={"error": str(e), "processing_time": time.time() - start_time}
                    )
                ))
            raise