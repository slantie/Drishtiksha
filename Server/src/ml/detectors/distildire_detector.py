# src/ml/models/distildire_detector.py

import os
import time
import torch
import logging
import tempfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop
import torchvision.transforms.functional as TF

from src.ml.base import BaseModel, AnalysisResult
from src.app.schemas import ImageAnalysisResult
from src.config import DistilDIREv1Config
from src.ml.exceptions import MediaProcessingError, InferenceError
from src.ml.event_publisher import event_publisher
from src.ml.schemas import ProgressEvent, EventData

# --- Guided Diffusion Imports ---
from src.ml.dependencies.guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults, dict_parse
from src.ml.dependencies.guided_diffusion.respace import SpacedDiffusion
from src.ml.architectures.distildire_resnet import create_distildire_model

logger = logging.getLogger(__name__)

class DistilDIREDetectorV1(BaseModel):
    """
    Image deepfake detector using Diffusion Reconstruction Error (DIRE).
    """
    config: DistilDIREv1Config

    def __init__(self, config: DistilDIREv1Config):
        super().__init__(config)
        self.adm_model: torch.nn.Module = None
        self.diffusion: SpacedDiffusion = None
        self.transform: Compose

    def load(self) -> None:
        start_time = time.time()
        try:
            logger.info(f"Loading detector model from: {self.config.model_path}")
            self.model = create_distildire_model(self.config.model_dump())
            state_dict = torch.load(self.config.model_path, map_location=self.device)['model']
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Loading ADM dependency model from: {self.config.adm_model_path}")
            adm_config = model_and_diffusion_defaults()
            adm_config.update(self.config.adm_config)
            
            self.adm_model, self.diffusion = create_model_and_diffusion(
                **dict_parse(adm_config, model_and_diffusion_defaults().keys())
            )
            self.adm_model.load_state_dict(torch.load(self.config.adm_model_path, map_location="cpu"))
            self.adm_model.to(self.device)
            if adm_config['use_fp16']:
                self.adm_model.convert_to_fp16()
            self.adm_model.eval()
            
            self.transform = Compose([
                Resize(self.config.image_size, antialias=True),
                CenterCrop(self.config.image_size)
            ])

            load_time = time.time() - start_time
            logger.info(f"✅ Loaded Model: '{self.config.model_name}' and its dependencies | Device: '{self.device}' | Time: {load_time:.2f}s.")

        except Exception as e:
            logger.error(f"Failed to load model '{self.config.model_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model '{self.config.model_name}'") from e

    def _get_first_step_noise(self, img_tensor: torch.Tensor) -> torch.Tensor:
        t = torch.zeros(img_tensor.shape[0],).long().to(self.device)
        model_kwargs = {}
        
        eps = self.diffusion.ddim_reverse_sample_only_eps(
            self.adm_model, x=img_tensor, t=t,
            clip_denoised=self.config.adm_config['clip_denoised'],
            model_kwargs=model_kwargs, eta=0.0
        )
        return eps

    def _save_tensor_as_image(self, tensor: torch.Tensor, file_path: str):
        """
        FIXED: Normalizes a tensor, transposes its dimensions, and saves it as a PNG image.
        """
        # --- START FIX ---
        tensor_np = tensor.squeeze().detach().cpu().numpy()
        
        # Transpose from (C, H, W) to (H, W, C) for image libraries
        if tensor_np.ndim == 3 and tensor_np.shape[0] in [1, 3, 4]:
             tensor_np = np.transpose(tensor_np, (1, 2, 0))

        # Normalize to 0-255 uint8 range for saving
        normalized = (tensor_np - tensor_np.min()) / (tensor_np.max() - tensor_np.min() + 1e-6)
        image_to_save = (normalized * 255).astype(np.uint8)

        # Use PIL for more robust image saving
        Image.fromarray(image_to_save).save(file_path, format='PNG')
        # --- END FIX ---
        logger.info(f"Saved visualization tensor to: {file_path}")

    def _generate_heatmap(self, feature_map: torch.Tensor, target_size: tuple) -> list:
        """Generates a heatmap from the model's feature map."""
        heatmap = torch.mean(feature_map, dim=1, keepdim=False).squeeze()
        heatmap_np = heatmap.detach().cpu().numpy()
        
        heatmap_resized = cv2.resize(heatmap_np, dsize=target_size, interpolation=cv2.INTER_CUBIC)
        
        heatmap_normalized = (heatmap_resized - np.min(heatmap_resized)) / (np.max(heatmap_resized) - np.min(heatmap_resized) + 1e-6)
        
        return heatmap_normalized.tolist()

    def analyze(self, media_path: str, generate_visualizations: bool = False, **kwargs) -> AnalysisResult:
        """
        Analyze an image for deepfake detection using DIRE (Diffusion Reconstruction Error).
        
        Args:
            media_path: Path to the media file
            generate_visualizations: If True, generate DIRE noise map visualization. Defaults to False.
            **kwargs: Additional arguments (media_id, user_id, etc.)
        """
        start_time = time.time()
        media_id = kwargs.get("media_id")
        user_id = kwargs.get("user_id")

        def publish(event: str, message: str, progress: int = 0, total: int = 100):
            if media_id and user_id:
                event_publisher.publish(ProgressEvent(
                    media_id=media_id, user_id=user_id, event="FRAME_ANALYSIS_PROGRESS", message=message,
                    data=EventData(
                        model_name=self.config.model_name, progress=progress, total=total,
                        details={"phase": event.lower()}
                    )
                ))

        try:
            publish("PREPROCESSING", "Preprocessing image", 0, 100)
            image = Image.open(media_path).convert("RGB")
            width, height = image.size
        except Exception as e:
            raise MediaProcessingError(f"Failed to open or process image at {media_path}. Error: {e}")

        visualization_path = None
        try:
            img_tensor = TF.to_tensor(image) * 2 - 1
            img_tensor = self.transform(img_tensor).unsqueeze(0).to(self.device)

            with torch.no_grad():
                publish("DIRE_GENERATION", "Generating DIRE noise map", 25, 100)
                eps_tensor = self._get_first_step_noise(img_tensor)

                # Only generate visualization if explicitly requested
                if generate_visualizations:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        visualization_path = tmp.name
                    self._save_tensor_as_image(eps_tensor, visualization_path)
                else:
                    logger.info(f"[{self.config.model_name}] Skipping DIRE map visualization generation (generate_visualizations=False)")
                
                publish("DETECTION", "Running detector model", 75, 100)
                combined_input = torch.cat([img_tensor, eps_tensor], dim=1)
                output = self.model(combined_input)
                prob_fake = torch.sigmoid(output['logit']).item()

            prediction = "FAKE" if prob_fake >= 0.5 else "REAL"
            confidence = prob_fake if prediction == "FAKE" else 1 - prob_fake
            
            feature_map = output.get('feature')
            heatmap_scores = self._generate_heatmap(feature_map, target_size=(width, height)) if feature_map is not None else None
            
            processing_time = time.time() - start_time
            
            publish("ANALYSIS_COMPLETE", f"Analysis complete: {prediction}", 100, 100)
            
            return ImageAnalysisResult(
                prediction=prediction,
                confidence=confidence,
                processing_time=processing_time,
                dimensions={"width": width, "height": height},
                heatmap_scores=heatmap_scores,
                visualization_path=visualization_path
            )

        except Exception as e:
            publish("ANALYSIS_FAILED", f"Error during analysis: {e}")
            logger.error(f"An error occurred during inference for {self.config.model_name}: {e}", exc_info=True)
            if visualization_path and os.path.exists(visualization_path):
                os.remove(visualization_path)
            raise InferenceError(f"An unexpected error occurred during analysis for {self.config.model_name}.")
