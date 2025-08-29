# src/ml/registry.py

import time
import inspect
import logging
import pkgutil
import importlib
from pathlib import Path
from typing import Dict, Type

from src.config import Settings, ModelConfig
from src.ml.base import BaseModel

logger = logging.getLogger(__name__)


class ModelRegistryError(Exception):
    """Custom exception for errors related to the model registry."""
    pass


class ModelManager:
    """
    REFACTORED automated, production-grade model manager.

    This class handles the automatic discovery, eager loading, and accessing
    of all ML models based on the application's central configuration.
    """
    def __init__(self, settings: Settings):
        self._models: Dict[str, BaseModel] = {}
        self.model_configs: Dict[str, ModelConfig] = settings.models
        
        # REFACTOR: The registry is now built dynamically at runtime.
        self._registry: Dict[str, Type[BaseModel]] = self._discover_models()
        
        logger.info(f"ModelManager initialized. Discovered {len(self._registry)} model classes.")
        logger.info(f"Active models from config: {list(self.model_configs.keys())}")

    def _discover_models(self) -> Dict[str, Type[BaseModel]]:
        """
        Automatically discovers all BaseModel subclasses in the `src.ml.models` package.
        
        This eliminates the need for manual registration, making the system
        extensible and less error-prone. A developer only needs to create the new
        model file, and the registry will find it automatically.
        """
        registry: Dict[str, Type[BaseModel]] = {}
        models_package_path = Path(__file__).parent / "models"
        
        # Dynamically import all modules within the `src.ml.models` directory.
        for (_, module_name, _) in pkgutil.iter_modules([str(models_package_path)]):
            module = importlib.import_module(f"src.ml.models.{module_name}")
            
            # Find all classes within the module that are subclasses of BaseModel.
            for class_name, cls in inspect.getmembers(module, inspect.isclass):
                if issubclass(cls, BaseModel) and cls is not BaseModel:
                    logger.debug(f"Discovered model class: {class_name}")
                    # The key is the class name string, which must match `class_name` in config.yaml
                    registry[class_name] = cls
                    
        return registry

    def load_models(self):
        """
        Eagerly loads all models specified as "active" in the configuration.

        This method is called once at server startup to ensure all models are
        loaded into memory and ready for inference before the server starts
        accepting traffic, preventing a "cold start" delay for the first user.
        """
        logger.info("Starting eager loading of all active models...")
        for model_name, model_config in self.model_configs.items():
            try:
                start_time = time.monotonic()
                class_name = model_config.class_name
                
                model_class = self._registry.get(class_name)
                if not model_class:
                    raise ModelRegistryError(
                        f"Model class '{class_name}' for model '{model_name}' not found in the registry. "
                        f"Ensure the class name in config.yaml matches the Python class name."
                    )
                
                logger.info(f"Loading model '{model_name}' (Class: {class_name})...")
                instance = model_class(model_config)
                instance.load() # This is the call that loads weights into GPU/CPU memory.
                
                self._models[model_name] = instance
                
                load_time = time.monotonic() - start_time
                logger.info(f"✅ Successfully loaded '{model_name}' in {load_time:.2f}s.")
                
            except Exception as e:
                # If any model fails to load, it's a critical error.
                logger.critical(f"❌ FATAL: Failed to load model '{model_name}'. Server startup will be aborted.", exc_info=True)
                raise ModelRegistryError(f"Could not load model '{model_name}'.") from e
        
        logger.info(f"✅ All {len(self._models)} active models have been loaded successfully.")

    def get_model(self, name: str) -> BaseModel:
        """
        Retrieves a pre-loaded model instance from the cache.

        Args:
            name: The name of the model to retrieve (e.g., "SIGLIP-LSTM-V3").

        Returns:
            An instance of a BaseModel subclass.

        Raises:
            KeyError: If the requested model is not loaded (i.e., not active or failed to load).
        """
        if name not in self._models:
            raise KeyError(
                f"Model '{name}' is not loaded. It might not be in ACTIVE_MODELS or it failed to load at startup."
            )
        return self._models[name]

    def get_available_models(self) -> list[str]:
        """Returns a list of all ACTIVE model names from the config."""
        return list(self.model_configs.keys())

    def get_loaded_model_names(self) -> list[str]:
        """Returns a list of names of models currently loaded in memory."""
        return list(self._models.keys())
    
    def get_active_model_configs(self) -> Dict[str, ModelConfig]:
        """Returns the configuration objects for all active models."""
        return self.model_configs