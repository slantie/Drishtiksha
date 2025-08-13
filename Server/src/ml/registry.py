# src/ml/registry.py

from typing import Dict, Type
from src.ml.base import BaseModel
from src.ml.models.lstm_detector import LSTMDetector

# This dictionary maps class names from config.yaml to actual Python classes.
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "LSTMDetector": LSTMDetector,
    # "NewAwesomeDetector": NewAwesomeDetector, # Example for the future
}

class ModelManager:
    """A manager to handle loading and accessing ML models on demand."""

    # --- THE FIX IS HERE ---
    # We add the __init__ method to correctly initialize the class instance.
    def __init__(self, model_configs: Dict):
        """
        Initializes the ModelManager.

        Args:
            model_configs: The 'models' dictionary from the main config file.
        """
        self._models: Dict[str, BaseModel] = {} # This will cache loaded models
        self.model_configs = model_configs       # Store the configuration

    def get_model(self, name: str) -> BaseModel:
        """Loads a model if not already cached, then returns it."""
        if name not in self._models:
            print(f"Model '{name}' not in cache. Initializing...")
            if name not in self.model_configs:
                raise ValueError(f"Configuration for model '{name}' not found in config.yaml.")
            
            config = self.model_configs[name]
            class_name = config.get("class_name")
            
            if class_name not in MODEL_REGISTRY:
                raise ValueError(f"Model class '{class_name}' not found in MODEL_REGISTRY.")

            model_class = MODEL_REGISTRY[class_name]
            config['name'] = name # Add the model's name to its own config dict
            instance = model_class(config)
            instance.load() # This is where weights are loaded
            self._models[name] = instance
        
        return self._models[name]