# src/ml/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseModel(ABC):
    """Abstract Base Class for all deepfake detection models."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.processor = None
        self.device = "cpu"

    @abstractmethod
    def load(self):
        """
        Loads the model and processor into memory.
        This method should set self.model, self.processor, and self.device.
        """
        pass

    @abstractmethod
    def predict(self, video_path: str) -> Dict[str, Any]:
        """
        Performs inference on a single video file.
        
        Returns:
            A dictionary containing at least 'prediction', 'confidence', and 'processing_time'.
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """Returns metadata about the model."""
        return {
            "model_name": self.config.get("name", "Unknown"),
            "description": self.config.get("description", "No description provided."),
            "model_path": self.config.get("model_path"),
            "processor_path": self.config.get("processor_path"),
            "device": self.device,
        }