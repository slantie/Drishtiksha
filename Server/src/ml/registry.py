# src/ml/registry.py

import logging
import string
from typing import Dict, Type

from src.config import Settings
from src.ml.base import BaseModel
from src.ml.models.siglip_lstm_detector import SiglipLSTMV1, SiglipLSTMV3, SiglipLSTMV4
from src.ml.models.color_cues_detector import ColorCuesLSTMV1
from src.ml.models.efficientnet_detector import EfficientNetB7Detector

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "SIGLIP-LSTM-V1": SiglipLSTMV1,
    "SIGLIP-LSTM-V3": SiglipLSTMV3,
    "SIGLIP-LSTM-V4": SiglipLSTMV4,
    "COLOR-CUES-LSTM-V1": ColorCuesLSTMV1,
    "EFFICIENTNET-B7-V1": EfficientNetB7Detector
}

class ModelManager:
    """
    Handles the loading, caching, and accessing of ML models based on the
    application's central configuration.
    """

    def __init__(self, settings: Settings):
        """
        Initializes the ModelManager with the application settings.

        Args:
            settings: The global, type-validated settings object.
        """
        self._models: Dict[str, BaseModel] = {}
        self.model_configs = settings.models
        models = ", ".join(name.upper() for name in self.model_configs.keys())
        logger.info(f"✅ Available Models: {models}")

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

            # Print model loading info to terminal
            print("=" * 80)
            print("🚀 LOADING MODEL")
            print("=" * 80)
            print(f"🤖 Model Name: {name}")
            print(f"🏷️  Model Class: {model_config.class_name}")
            print(f"🖥️  Device: {model_config.device}")
            print(f"📁 Model Path: {model_config.model_path}")
            print("⏳ Loading...")
            print("=" * 80)

            instance = model_class(model_config)
            instance.load()
            self._models[name] = instance
            
            # Print success message
            print("=" * 80)
            print("✅ MODEL LOADED SUCCESSFULLY")
            print("=" * 80)
            print(f"🤖 Model Name: {name}")
            print(f"🖥️  Device: {model_config.device}")
            print("🎯 Ready for inference!")
            print("=" * 80)

        return self._models[name]

    def get_available_models(self) -> list[str]:
        """Returns a list of all configured model names."""
        return list(self.model_configs.keys())