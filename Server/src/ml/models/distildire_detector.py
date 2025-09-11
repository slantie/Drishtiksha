# src/ml/models/distildire_detector.py

import os
import time
import torch
import logging
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop
import torchvision.transforms.functional as TF

from src.ml.base import BaseModel, AnalysisResult
from src.app.schemas import ImageAnalysisResult
from src.config import DistilDIREv1Config
from src.ml.exceptions import MediaProcessingError, InferenceError

# --- Guided Diffusion Imports (now part of the project structure) ---
from src.ml.dependencies.guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults, dict_parse
from src.ml.dependencies.guided_diffusion.respace import SpacedDiffusion
from src.ml.architectures.distildire_resnet import create_distildire_model

logger = logging.getLogger(__name__)

class DistilDIREDetectorV1(BaseModel):
    """
    Image deepfake detector using Diffusion Reconstruction Error (DIRE).
    
    This model uses a two-stage process:
    1. A pre-trained ADM diffusion model calculates a one-step noise map ('eps').
    2. A fine-tuned ResNet-50 detector takes both the image and the noise map
       as input to make the final prediction.
    """
    config: DistilDIREv1Config

    def __init__(self, config: DistilDIREv1Config):
        super().__init__(config)
        self.adm_model: torch.nn.Module = None
        self.diffusion: SpacedDiffusion = None
        self.transform: Compose

    def load(self) -> None:
        """
        Loads both the main detector model and the required ADM diffusion model.
        """
        start_time = time.time()
        try:
            # 1. Load the main DistilDIRE detector model
            logger.info(f"Loading detector model from: {self.config.model_path}")
            self.model = create_distildire_model(self.config.model_dump())
            state_dict = torch.load(self.config.model_path, map_location=self.device)['model']
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            # 2. Load the ADM model required for noise map generation
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
            
            # 3. Define image transformations
            self.transform = Compose([
                Resize(self.config.image_size, antialias=True),
                CenterCrop(self.config.image_size)
            ])

            load_time = time.time() - start_time
            logger.info(f"✅ Loaded Model: '{self.config.class_name}' and its dependencies | Device: '{self.device}' | Time: {load_time:.2f}s.")

        except Exception as e:
            logger.error(f"Failed to load model '{self.config.class_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model '{self.config.class_name}'") from e

    def _get_first_step_noise(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Uses the ADM model to calculate the one-step reverse diffusion noise (eps).
        """
        t = torch.zeros(img_tensor.shape[0],).long().to(self.device)
        model_kwargs = {}
        
        eps = self.diffusion.ddim_reverse_sample_only_eps(
            self.adm_model,
            x=img_tensor,
            t=t,
            clip_denoised=self.config.adm_config['clip_denoised'],
            model_kwargs=model_kwargs,
            eta=0.0
        )
        return eps

    def analyze(self, media_path: str, **kwargs) -> AnalysisResult:
        """
        The unified entry point for performing a complete analysis on an image file.
        """
        start_time = time.time()
        
        try:
            image = Image.open(media_path).convert("RGB")
            width, height = image.size
        except Exception as e:
            raise MediaProcessingError(f"Failed to open or process image at {media_path}. Error: {e}")

        try:
            # --- Preprocessing ---
            img_tensor = TF.to_tensor(image) * 2 - 1  # Normalize to [-1, 1]
            img_tensor = self.transform(img_tensor).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # --- Step 1: Generate the 'eps' noise map ---
                eps_tensor = self._get_first_step_noise(img_tensor)
                
                # --- Step 2: Concatenate image and noise map ---
                combined_input = torch.cat([img_tensor, eps_tensor], dim=1)

                # --- Step 3: Get prediction from the detector ---
                output = self.model(combined_input)
                prob_fake = torch.sigmoid(output['logit']).item()

            prediction = "FAKE" if prob_fake >= 0.5 else "REAL"
            confidence = prob_fake if prediction == "FAKE" else 1 - prob_fake
            
            processing_time = time.time() - start_time
            
            # --- Assemble and return the final, standardized result ---
            return ImageAnalysisResult(
                prediction=prediction,
                confidence=confidence,
                processing_time=processing_time,
                dimensions={"width": width, "height": height}
            )

        except Exception as e:
            logger.error(f"An error occurred during inference for {self.config.class_name}: {e}", exc_info=True)
            raise InferenceError(f"An unexpected error occurred during analysis for {self.config.class_name}.")