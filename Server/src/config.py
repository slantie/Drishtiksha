# src/config.py

import logging
import yaml
from pydantic import BaseModel, Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, Union, Annotated, Literal

logger = logging.getLogger(__name__)

# A dedicated Pydantic model for the SigLIP architecture details
class SiglipArchitectureConfig(BaseModel):
    base_model_path: str
    lstm_hidden_size: int
    lstm_num_layers: int
    num_classes: int

# New architecture config for V4 to include dropout
class SiglipArchitectureV4Config(SiglipArchitectureConfig):
    dropout_rate: float = 0.5

# Base model for shared configuration fields
class BaseModelConfig(BaseModel):
    class_name: str
    description: str
    model_path: str
    device: str = "cuda"  # Default value, will be overridden by environment variable

# Specific configuration model for SIGLIP-LSTM-V1 - Legacy Model
class SiglipLSTMv1Config(BaseModelConfig):
    class_name: Literal["SIGLIP-LSTM-V1"]
    processor_path: str
    num_frames: int
    model_definition: SiglipArchitectureConfig

# Specific configuration model for SIGLIP-LSTM-V3
class SiglipLSTMv3Config(BaseModelConfig):
    class_name: Literal["SIGLIP-LSTM-V3"]
    processor_path: str
    num_frames: int
    rolling_window_size: int
    model_definition: SiglipArchitectureConfig

# New configuration model for SIGLIP-LSTM-V4
class SiglipLSTMv4Config(BaseModelConfig):
    class_name: Literal["SIGLIP-LSTM-V4"]
    processor_path: str
    num_frames: int
    rolling_window_size: int
    model_definition: SiglipArchitectureV4Config

# Specific configuration model for COLOR-CUES-LSTM-V1
class ColorCuesConfig(BaseModelConfig):
    class_name: Literal["COLOR-CUES-LSTM-V1"]
    dlib_model_path: str
    sequence_length: int
    frames_per_video: int
    histogram_bins: int
    landmark_margin: int
    rolling_window_size: int
    hidden_size: int
    dropout: float

# Configuration model for EFFICIENTNET-B7-V1
class EfficientNetB7Config(BaseModelConfig):
    class_name: Literal["EFFICIENTNET-B7-V1"]
    encoder: str
    input_size: int

# A Discriminated Union to allow only the available models to be served
ModelConfig = Annotated[
    Union[
        ColorCuesConfig,
        SiglipLSTMv1Config,
        SiglipLSTMv3Config,
        SiglipLSTMv4Config,
        EfficientNetB7Config
    ],
    Field(discriminator="class_name"),
]

# The main Settings class that loads from both .env and config.yaml
class Settings(BaseSettings):
    api_key: SecretStr
    default_model_name: str
    project_name: str
    device: str
    active_models: str = "SIGLIP-LSTM-V1,SIGLIP-LSTM-V3,COLOR-CUES-LSTM-V1"  # Default to all models
    models: Dict[str, ModelConfig]

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

    @property
    def active_model_list(self) -> list[str]:
        """Returns a list of active model names."""
        return [model.strip() for model in self.active_models.split(',')]

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Settings":
        """Factory method to load settings from a YAML file and merge with .env."""
        import os
        
        # PRIORITY 1: Load .env file first
        try:
            from dotenv import load_dotenv
            load_dotenv()
            logger.info("🔧 Loaded environment variables from .env file")
        except ImportError:
            logger.warning("⚠️ python-dotenv not available, using system environment variables only")
        
        # PRIORITY 2: Get device from environment variable, fallback to 'cpu'
        env_device = os.getenv('DEVICE', 'cpu').lower()
        logger.info(f"🔧 Device setting from environment: '{env_device}'")
        
        try:
            with open(yaml_path, 'r') as file:
                yaml_config = yaml.safe_load(file)
                if not yaml_config:
                    raise ValueError("YAML file is empty or invalid.")
        except FileNotFoundError:
            raise RuntimeError(f"Configuration file '{yaml_path}' not found.")
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing YAML file '{yaml_path}': {e}")

        # PRIORITY 3: Override global device setting with environment variable
        yaml_config['device'] = env_device
        
        # PRIORITY 4: Override ALL model device settings with environment variable
        if yaml_config.get('models'):
            for model_name, model_config in yaml_config['models'].items():
                if isinstance(model_config, dict):
                    model_config['device'] = env_device
                    logger.info(f"🔧 Forcing device '{env_device}' for model '{model_name}'")

        # Create a temporary instance to get other environment variables
        temp_settings = cls(**yaml_config)
        
        # Filter models based on active_models configuration
        active_model_list = [model.strip() for model in temp_settings.active_models.split(',')]
        filtered_models = {
            model_name: model_config 
            for model_name, model_config in yaml_config.get('models', {}).items()
            if model_name in active_model_list
        }
        
        # Update yaml_config with filtered models
        yaml_config['models'] = filtered_models
        
        # Log which models are being loaded
        if filtered_models:
            active_names = ', '.join(filtered_models.keys())
            logger.info(f"🔧 Loading active models: {active_names}")
        else:
            logger.warning("⚠️ No active models configured!")

        return cls(**yaml_config)

try:
    settings = Settings.from_yaml("configs/config.yaml")
except (RuntimeError, ValueError, ValidationError) as e:
    logger.critical(f"❌ FATAL: Could not load configuration. {e}")
    exit(1)