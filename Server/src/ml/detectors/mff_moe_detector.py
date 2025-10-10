import time
import torch
import logging
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import List, Tuple, Optional

from src.ml.base import BaseModel, AnalysisResult
from src.app.schemas import ImageAnalysisResult, VideoAnalysisResult, FramePrediction
from src.config import MFFMoEV1Config # We will create this in the next step
from src.ml.exceptions import MediaProcessingError, InferenceError
from src.ml.utils import extract_frames
from src.ml.architectures.mff_moe import MFF_MoE

logger = logging.getLogger(__name__)

class MFFMoEDetectorV1(BaseModel):
    """
    Handler for the MFF-MoE (Multi-task Feature Fusion Mixture-of-Experts) model.
    This model can analyze both single images and videos on a per-frame basis.
    """
    config: MFFMoEV1Config

    def __init__(self, config: MFFMoEV1Config):
        super().__init__(config)
        self.transform: Optional[transforms.Compose] = None

    def load(self) -> None:
        """
        Loads the MFF-MoE model using its custom loading mechanism.
        The model path in the config should point to the DIRECTORY containing
        'weight.pth' and 'ema.state'.
        """
        start_time = time.time()
        try:
            self.model = MFF_MoE(pretrained=False)  # Initialize without pretrained weights
            # The model's custom 'load' method handles loading both state dicts
            self.model.load(path=str(self.config.model_path))
            self.model.to(self.device)
            self.model.eval()

            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize((512, 512), antialias=True),
            ])
            load_time = time.time() - start_time
            logger.info(f"✅ Loaded Model: '{self.config.model_name}' | Device: '{self.device}' | Time: {load_time:.2f}s.")
        except Exception as e:
            logger.error(f"Failed to load model '{self.config.model_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model '{self.config.model_name}'") from e

    def _analyze_image(self, image: Image.Image) -> Tuple[float, Optional[str]]:
        """Analyzes a single PIL image and returns the fake probability."""
        try:
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                prob_fake = self.model(image_tensor).item()
            return prob_fake, None
        except Exception as e:
            logger.error(f"Error during single image inference for {self.config.model_name}: {e}")
            return 0.5, "Inference on image failed."

    def analyze(self, media_path: str, **kwargs) -> AnalysisResult:
        """
        Unified analysis entry point. Dispatches to image or video analysis
        based on the media type.
        """
        from src.ml.utils import get_media_type
        media_type = get_media_type(media_path)
        
        if media_type == "IMAGE":
            return self._run_image_analysis(media_path)
        elif media_type == "VIDEO":
            return self._run_video_analysis(media_path)
        else:
            raise MediaProcessingError(f"Unsupported media type for MFF-MoE model: {media_type}")

    def _run_image_analysis(self, image_path: str) -> ImageAnalysisResult:
        """Performs analysis on a single image file."""
        start_time = time.time()
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise MediaProcessingError(f"Could not open image file: {e}")

        prob_fake, note = self._analyze_image(image)
        prediction = "FAKE" if prob_fake >= 0.5 else "REAL"
        confidence = prob_fake if prediction == "FAKE" else 1 - prob_fake

        return ImageAnalysisResult(
            prediction=prediction,
            confidence=confidence,
            processing_time=time.time() - start_time,
            note=note,
            dimensions={"width": image.width, "height": image.height}
        )

    def _run_video_analysis(self, video_path: str) -> VideoAnalysisResult:
        """Performs frame-by-frame analysis on a video file."""
        start_time = time.time()
        frame_predictions: List[FramePrediction] = []
        frame_scores: List[float] = []
        
        try:
            # Sample up to 50 frames evenly spaced throughout the video
            frame_generator = extract_frames(video_path, num_frames=self.config.video_frames_to_sample)
            frames = list(frame_generator)
            
            if not frames:
                raise MediaProcessingError("Could not extract any frames from the video.")

            for i, frame_pil in enumerate(frames):
                prob_fake, _ = self._analyze_image(frame_pil)
                frame_scores.append(prob_fake)
                frame_predictions.append(
                    FramePrediction(
                        index=i,
                        score=prob_fake,
                        prediction="FAKE" if prob_fake >= 0.5 else "REAL"
                    )
                )
            
            if not frame_scores:
                raise InferenceError("Frame analysis yielded no results.")

            # Aggregate results
            avg_score = np.mean(frame_scores)
            prediction = "FAKE" if avg_score >= 0.5 else "REAL"
            confidence = avg_score if prediction == "FAKE" else 1 - avg_score
            
            return VideoAnalysisResult(
                prediction=prediction,
                confidence=confidence,
                processing_time=time.time() - start_time,
                frames_analyzed=len(frames),
                frame_predictions=frame_predictions,
                metrics={"final_average_score": avg_score}
            )

        except (MediaProcessingError, InferenceError) as e:
            raise e # Re-raise known errors
        except Exception as e:
            logger.error(f"Unexpected error during video analysis for {self.config.model_name}: {e}", exc_info=True)
            raise InferenceError(f"An unexpected error occurred during video analysis.")
