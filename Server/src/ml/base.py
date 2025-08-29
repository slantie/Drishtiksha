# src/ml/base.py

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Union

from src.config import ModelConfig
# REFACTOR: We import the new, unified schemas that we will define next.
from src.app.schemas import VideoAnalysisResult, AudioAnalysisResult

logger = logging.getLogger(__name__)

# REFACTOR: Define a type hint for our standardized analysis output.
# This ensures that any model's analyze() method will return one of our approved data structures.
AnalysisResult = Union[VideoAnalysisResult, AudioAnalysisResult]


class BaseModel(ABC):
    """
    REFACTORED Abstract Base Class for all machine learning models.

    This class defines a clean, simple, and powerful contract that all model
    implementations must adhere to. It is designed to be media-type agnostic,
    supporting video, audio, images, and more.
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

    @abstractmethod
    def load(self) -> None:
        """
        Loads the model, weights, and any necessary processors into memory.
        This method is called once at server startup for each active model.
        It should set self.model and any other required components.
        """
        pass

    @abstractmethod
    def analyze(self, media_path: str, **kwargs) -> AnalysisResult:
        """
        Performs a comprehensive analysis on the given media file.

        This is the single, unified entry point for all model inference. The method
        should perform all necessary processing and return a structured Pydantic
        model containing the complete results. The structure of the return object
        (e.g., VideoAnalysisResult vs. AudioAnalysisResult) depends on the model's media type.

        Args:
            media_path (str): The local file path to the media to be analyzed.
            **kwargs: Catches additional parameters that might be sent from the API,
                      such as 'video_id' or 'user_id' for event publishing.

        Returns:
            An instance of a Pydantic model (e.g., VideoAnalysisResult) that
            encapsulates all the analysis data.
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """Returns metadata about the model, derived from its config."""
        return {
            "model_name": self.config.description,
            "class_name": self.config.class_name,
            "model_path": self.config.model_path,
            "device": self.device,
        }