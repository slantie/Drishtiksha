# src/ml/base.py

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

# Import our new, strongly-typed config models
from src.config import ModelConfig

# Set up a logger for the ML module
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract Base Class for all deepfake detection models.

    This class defines a strict contract that all model implementations must adhere to,
    ensuring they are compatible with all API endpoints.
    """

    def __init__(self, config: ModelConfig):
        """
        Initializes the model with its specific, type-validated configuration.

        Args:
            config (ModelConfig): A Pydantic model containing the configuration
                                  for this specific model instance.
        """
        self.config = config
        self.model: Any = None
        self.processor: Any = None
        self.device = config.device
        # logger.info(f"Initializing model with config: {config.model_dump()}")

    @abstractmethod
    def load(self) -> None:
        """
        Loads the model, weights, and any necessary processors into memory.
        This method should set self.model and self.processor.
        """
        pass

    @abstractmethod
    def predict(self, video_path: str) -> Dict[str, Any]:
        """
        Performs a quick analysis on a video file. Corresponds to the
        `/analyze` endpoint.

        Returns:
            A dictionary containing at least 'prediction', 'confidence',
            and 'processing_time'.
        """
        pass

    def predict_detailed(self, video_path: str) -> Dict[str, Any]:
        """
        Performs a detailed analysis with extra metrics. Corresponds to the
        `/analyze/detailed` endpoint.

        Can be overridden by subclasses for model-specific detailed output.
        By default, it calls the standard predict method.
        """
        raise NotImplementedError(
            f"Detailed analysis is not implemented for model '{self.config.class_name}'"
        )

    def predict_frames(self, video_path: str) -> Dict[str, Any]:
        """
        Performs a frame-by-frame analysis. Corresponds to the
        `/analyze/frames` endpoint.
        """
        raise NotImplementedError(
            f"Frame-by-frame analysis is not implemented for model '{self.config.class_name}'"
        )

    def predict_visual(self, video_path: str) -> str:
        """
        Performs analysis and returns a path to a visualized video output.
        Corresponds to the `/analyze/visualize` endpoint.

        Returns:
            The file path to the generated visualized video.
        """
        raise NotImplementedError(
            f"Visual analysis is not implemented for model '{self.config.class_name}'"
        )

    def get_info(self) -> Dict[str, Any]:
        """Returns metadata about the model, derived from its config."""
        # This method is now much cleaner thanks to the typed config
        return {
            "model_name": self.config.description,
            "class_name": self.config.class_name,
            "model_path": self.config.model_path,
            "device": self.device,
        }