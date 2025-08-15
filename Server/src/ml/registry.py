# src/ml/registry.py

import logging
from typing import Dict, Type

from src.config import Settings
from src.ml.base import BaseModel
from src.ml.models.siglip_lstm_detector import SiglipLSTMV1, SiglipLSTMV3
from src.ml.models.color_cues_detector import ColorCuesDetector

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "LSTMDetector": SiglipLSTMV1,
    "LSTMDetectorV3": SiglipLSTMV3,
    "ColorCuesDetector": ColorCuesDetector,
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
        logger.info(f"ModelManager initialized with models: {list(self.model_configs.keys())}")

    def get_model(self, name: str) -> BaseModel:
        """
        Loads a model if not already in the cache, then returns the instance.
        Models are loaded lazily (on first request).

        Args:
            name: The name of the model to retrieve (e.g., "siglip-lstm-v3").

        Returns:
            An instance of a BaseModel subclass.

        Raises:
            ValueError: If the requested model name is not found in the configuration.
        """
        if name not in self._models:
            logger.info(f"Model '{name}' not in cache. Loading...")
            
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
            logger.info(f"Model '{name}' successfully loaded and cached.")

        return self._models[name]

    def get_available_models(self) -> list[str]:
        """Returns a list of all configured model names."""
        return list(self.model_configs.keys())