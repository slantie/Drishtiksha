# Server/src/ml/detectors/cross_efficient_vit_detector.py

import torch
import cv2
import numpy as np
import logging
import time
from PIL import Image
from typing import List, Tuple, Optional
from torchvision.transforms import Compose, Normalize
from facenet_pytorch import MTCNN

from src.ml.base import BaseModel, AnalysisResult
from src.app.schemas import VideoAnalysisResult, ImageAnalysisResult, FramePrediction
from src.config import CrossEfficientViTConfig
from src.ml.exceptions import MediaProcessingError, InferenceError
from src.ml.utils import extract_frames

# Import the new architecture and the specific resize transform
from src.ml.architectures.cross_efficient_vit_arch import CrossEfficientViT
from src.ml.dependencies.efficientnet_pytorch.utils import get_same_padding_conv2d

# Import albumentations transforms from the new repository
# (Assuming you copy transforms/albu.py to a suitable location like src/ml/dependencies)
# For now, let's redefine the necessary part here for simplicity
from albumentations import Compose, Normalize, Resize 
from albumentations.pytorch import ToTensorV2


logger = logging.getLogger(__name__)

def aggregate_predictions(preds: List[float]) -> float:
    """ Aggregates frame/crop predictions for a single media file. """
    if not preds:
        return 0.0
    # A simple but effective heuristic: if any frame is highly suspicious,
    # the video is likely fake. Otherwise, average the suspicion.
    for pred_value in preds:
        if pred_value > 0.65:
            return pred_value
    return float(np.mean(preds))


class CrossEfficientViTDetector(BaseModel):
    config: CrossEfficientViTConfig

    def __init__(self, config: CrossEfficientViTConfig):
        super().__init__(config)
        self.face_detector: Optional[MTCNN] = None
        self.transform: Optional[Compose] = None

    def load(self) -> None:
        """Loads the CrossEfficientViT model and its dependencies."""
        start_time = time.time()
        try:
            # The model's config is nested under `model_definition`
            arch_config_dict = {'model': self.config.model_definition.model_dump(by_alias=True)}
            self.model = CrossEfficientViT(config=arch_config_dict)
            
            state_dict = torch.load(self.config.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            # Initialize the face detector
            self.face_detector = MTCNN(
                keep_all=True, device=self.device, select_largest=False,
                min_face_size=20, thresholds=[0.6, 0.7, 0.7]
            )

            # Define the image transformation pipeline
            self.transform = Compose([
                Resize(height=self.config.model_definition.image_size, width=self.config.model_definition.image_size),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

            load_time = time.time() - start_time
            logger.info(f"✅ Loaded Model: '{self.config.class_name}' | Device: '{self.device}' | Time: {load_time:.2f}s.")
        
        except Exception as e:
            logger.error(f"Failed to load model '{self.config.class_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model '{self.config.class_name}'") from e

    def _get_face_crops(self, frame: np.ndarray) -> List[np.ndarray]:
        """Detects and extracts all face crops from a single frame."""
        try:
            # Convert BGR (cv2) to RGB (PIL/facenet)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # The detector returns a list of cropped face tensors
            face_tensors = self.face_detector(frame_rgb, return_prob=False)
            
            if face_tensors is None:
                return []
            
            # Convert tensors back to numpy arrays for further processing/saving
            face_crops = []
            for face_tensor in face_tensors:
                face_np = face_tensor.permute(1, 2, 0).cpu().numpy()
                face_np = (face_np * 255).astype(np.uint8)
                face_crops.append(face_np)
            return face_crops
        except Exception as e:
            logger.warning(f"Face detection failed for a frame: {e}")
            return []

    def _predict_on_crops(self, crops: List[np.ndarray]) -> List[float]:
        """Runs the classifier on a list of face crops."""
        if not crops:
            return []

        transformed_crops = [self.transform(image=crop)['image'] for crop in crops]
        batch_tensor = torch.stack(transformed_crops).to(self.device)

        with torch.no_grad():
            logits = self.model(batch_tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy().flatten().tolist()
        
        return probabilities

    def _analyze_image(self, image_path: str) -> ImageAnalysisResult:
        """Performs analysis on a single image file."""
        start_time = time.time()
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise MediaProcessingError(f"Could not read image file: {image_path}")
            height, width = image.shape[:2]
        except Exception as e:
            raise MediaProcessingError(f"Failed to open or process image: {e}")

        face_crops = self._get_face_crops(image)
        if not face_crops:
            return ImageAnalysisResult(
                prediction="REAL", confidence=0.99, processing_time=time.time() - start_time,
                note="No face detected in the image.", dimensions={"width": width, "height": height}
            )

        crop_preds = self._predict_on_crops(face_crops)
        final_score = aggregate_predictions(crop_preds)
        prediction = "FAKE" if final_score > 0.5 else "REAL"
        confidence = final_score if prediction == "FAKE" else 1 - final_score
        
        return ImageAnalysisResult(
            prediction=prediction, confidence=confidence,
            processing_time=time.time() - start_time,
            dimensions={"width": width, "height": height}
        )

    def _analyze_video(self, video_path: str) -> VideoAnalysisResult:
        """Performs frame-by-frame analysis on a video file."""
        start_time = time.time()
        frame_predictions: List[FramePrediction] = []
        
        try:
            frame_generator = extract_frames(video_path, num_frames=30)
            frames = list(frame_generator)
            
            if not frames:
                raise MediaProcessingError("Could not extract any frames from the video.")

            all_crop_preds = []
            for i, frame_pil in enumerate(frames):
                frame_np = np.array(frame_pil)
                face_crops = self._get_face_crops(frame_np)
                
                if not face_crops:
                    frame_predictions.append(FramePrediction(index=i, score=0.0, prediction="REAL"))
                    continue

                crop_preds = self._predict_on_crops(face_crops)
                all_crop_preds.extend(crop_preds)
                
                frame_score = max(crop_preds) if crop_preds else 0.0
                frame_predictions.append(
                    FramePrediction(
                        index=i, score=frame_score,
                        prediction="FAKE" if frame_score > 0.5 else "REAL"
                    )
                )
            
            if not all_crop_preds:
                return VideoAnalysisResult(
                    prediction="REAL", confidence=0.99, processing_time=time.time() - start_time,
                    note="No faces were detected in any of the sampled frames.",
                    frames_analyzed=len(frames), frame_predictions=frame_predictions,
                    metrics={"final_average_score": 0.0}
                )

            final_score = aggregate_predictions(all_crop_preds)
            prediction = "FAKE" if final_score > 0.5 else "REAL"
            confidence = final_score if prediction == "FAKE" else 1 - final_score
            
            return VideoAnalysisResult(
                prediction=prediction, confidence=confidence,
                processing_time=time.time() - start_time,
                frames_analyzed=len(frames),
                frame_predictions=frame_predictions,
                metrics={"final_average_score": final_score}
            )
        except (MediaProcessingError, InferenceError) as e:
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during video analysis: {e}", exc_info=True)
            raise InferenceError("An unexpected error occurred during video analysis.")

    def analyze(self, media_path: str, **kwargs) -> AnalysisResult:
        """
        Unified analysis that dispatches based on the model's configuration.
        """
        if self.config.isVideo:
            return self._analyze_video(media_path)
        elif self.config.isImage:
            return self._analyze_image(media_path)
        else:
            raise NotImplementedError(
                f"Model {self.config.class_name} does not support video or image analysis."
            )