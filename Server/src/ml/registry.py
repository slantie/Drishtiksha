# src/ml/registry.py


import logging
from typing import Dict, Type
from src.ml.base import BaseModel
from src.config import Settings, ModelConfig
from src.ml.models.color_cues_detector import ColorCuesLSTMV1
from src.ml.models.eyeblink_detector import EyeblinkDetectorV1
from src.ml.models.efficientnet_detector import EfficientNetB7Detector
from src.ml.models.scattering_wave_detector import ScatteringWaveV1
from src.ml.models.siglip_lstm_detector import SiglipLSTMV1, SiglipLSTMV3, SiglipLSTMV4

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "EFFICIENTNET-B7-V1": EfficientNetB7Detector,
    "EYEBLINK-CNN-LSTM-V1": EyeblinkDetectorV1,
    "SIGLIP-LSTM-V4": SiglipLSTMV4,
    "SCATTERING-WAVE-V1": ScatteringWaveV1,
    "SIGLIP-LSTM-V3": SiglipLSTMV3,
    "SIGLIP-LSTM-V1": SiglipLSTMV1,
    "COLOR-CUES-LSTM-V1": ColorCuesLSTMV1
}

class ModelManager:
    """
    Handles the loading, caching, and accessing of ML models based on the
    application's central configuration.
    """

    def __init__(self, settings: Settings):
        self._models: Dict[str, BaseModel] = {}
        # This now correctly stores only the active model configurations
        self.model_configs = settings.models 
        logger.info(f"✅ ModelManager initialized for active models: {', '.join(self.model_configs.keys())}")

    def get_model(self, name: str) -> BaseModel:
        """
        Loads a model if not already in the cache, then returns the instance.
        Models are loaded lazily (on first request).

        Args:
            name: The name of the model to retrieve (e.g., "SIGLIP-LSTM-V3").

        Returns:
            An instance of a BaseModel subclass.

        Raises:
            ValueError: If the requested model name is not found in the configuration.
        """
        if name not in self._models:
            model_config = self.model_configs.get(name)
            if not model_config:
                available = list(self.model_configs.keys())
                raise ValueError(
                    f"Configuration for model '{name}' not found. Available models: {available}"
                )

            model_class = MODEL_REGISTRY.get(model_config.class_name)
            if not model_class:
                raise ValueError(
                    f"Model class '{model_config.class_name}' for model '{name}' not found in MODEL_REGISTRY."
                )

            instance = model_class(model_config)
            instance.load()
            self._models[name] = instance

        return self._models[name]

    def get_available_models(self) -> list[str]:
        """Returns a list of all ACTIVE model names."""
        return list(self.model_configs.keys())

    # FIX: Add a public method to get loaded model names, avoiding private access.
    def get_loaded_model_names(self) -> list[str]:
        """Returns a list of names of models currently loaded in memory."""
        return list(self._models.keys())
    
    # FIX: Add a public method to get the configurations of active models.
    def get_active_model_configs(self) -> Dict[str, ModelConfig]:
        """Returns the configuration objects for all active models."""
        return self.model_configs