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
        detectors_package_path = Path(__file__).parent / "detectors"

        # Dynamically import all modules within the `src.ml.detectors` directory.
        for (_, module_name, _) in pkgutil.iter_modules([str(detectors_package_path)]):
            if module_name == "__init__":
                continue
            
            module = importlib.import_module(f"src.ml.detectors.{module_name}")
            
            # Find all classes within the module that are subclasses of BaseModel.
            for class_name, cls in inspect.getmembers(module, inspect.isclass):
                if issubclass(cls, BaseModel) and cls is not BaseModel:
                    logger.debug(f"Discovered model class: {class_name}")
                    registry[class_name] = cls
                    
        return registry

    def load_model(self, model_name: str) -> BaseModel:
        """
        Lazy loads a single model on-demand.
        
        If the model is already loaded, returns the cached instance.
        If not loaded, loads it into memory and caches it.
        
        Args:
            model_name: The name of the model to load (e.g., "EFFICIENTNET-B7-V1")
            
        Returns:
            The loaded model instance
            
        Raises:
            ModelRegistryError: If model config not found or loading fails
        """
        # Return cached model if already loaded
        if model_name in self._models:
            logger.debug(f"Model '{model_name}' already loaded, returning cached instance.")
            return self._models[model_name]
        
        # Check if model is configured
        if model_name not in self.model_configs:
            available = list(self.model_configs.keys())
            raise ModelRegistryError(
                f"Model '{model_name}' not found in configuration. "
                f"Available models: {available}"
            )
        
        # Load the model
        model_config = self.model_configs[model_name]
        class_name = model_config.class_name
        
        model_class = self._registry.get(class_name)
        if not model_class:
            raise ModelRegistryError(
                f"Model class '{class_name}' for model '{model_name}' not found in the registry. "
                f"Ensure the class name in config.yaml matches the Python class name."
            )
        
        try:
            logger.info(f"🔄 Lazy loading model '{model_name}' (Class: {class_name})...")
            start_time = time.monotonic()
            
            # CRITICAL FIX: Inject the model_name into config before instantiation
            # This ensures progress events use the consistent model key (e.g., "EFFICIENTNET-B7-V1")
            # instead of the class name (e.g., "EfficientNetB7Detector")
            model_config.model_name = model_name
            
            instance = model_class(model_config)
            instance.load()  # Load weights into GPU/CPU memory
            
            self._models[model_name] = instance
            
            load_time = time.monotonic() - start_time
            logger.info(f"✅ Successfully loaded '{model_name}' in {load_time:.2f}s.")
            
            return instance
            
        except Exception as e:
            logger.error(f"❌ Failed to load model '{model_name}'.", exc_info=True)
            raise ModelRegistryError(f"Could not load model '{model_name}'.") from e
    
    def load_models(self):
        """
        Eagerly loads all models specified in the configuration.
        
        This method is used by the web server for startup preloading.
        CLI should NOT call this - it should use lazy loading via get_model() instead.
        """
        logger.info("Starting eager loading of all configured models...")
        for model_name in self.model_configs.keys():
            try:
                self.load_model(model_name)
            except Exception as e:
                # For server startup, model load failures are critical
                logger.critical(f"❌ FATAL: Failed to load model '{model_name}'. Server startup will be aborted.", exc_info=True)
                raise ModelRegistryError(f"Could not load model '{model_name}'.") from e
        
        logger.info(f"✅ All {len(self._models)} models have been loaded successfully.")

    def get_model(self, name: str) -> BaseModel:
        """
        Retrieves a model instance, loading it lazily if not already loaded.
        
        This enables on-demand loading - models are only loaded when actually needed.
        Perfect for CLI usage where you don't want to load all models upfront.

        Args:
            name: The name of the model to retrieve (e.g., "SIGLIP-LSTM-V3").

        Returns:
            An instance of a BaseModel subclass.

        Raises:
            ModelRegistryError: If the requested model config is not found or loading fails.
        """
        return self.load_model(name)

    def get_available_models(self) -> list[str]:
        """Returns a list of all ACTIVE model names from the config."""
        return list(self.model_configs.keys())

    def get_loaded_model_names(self) -> list[str]:
        """Returns a list of names of models currently loaded in memory."""
        return list(self._models.keys())
    
    def is_model_loaded(self, name: str) -> bool:
        """
        Check if a model is currently loaded in memory.
        
        This does NOT trigger loading - just checks the cache.
        
        Args:
            name: The model name to check
            
        Returns:
            True if the model is loaded, False otherwise
        """
        return name in self._models
    
    def get_active_model_configs(self) -> Dict[str, ModelConfig]:
        """Returns the configuration objects for all active models."""
        return self.model_configs