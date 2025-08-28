# src/config.py

import yaml
import logging
from typing import Dict, Tuple, Union, Annotated, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel, Field, SecretStr, ValidationError

# Use __name__ for a module-specific logger, which is a standard best practice.
logger = logging.getLogger(__name__)

# A dedicated Pydantic model for the SigLIP architecture details
class SiglipArchitectureConfig(BaseModel):
    base_model_path: str
    lstm_hidden_size: int
    lstm_num_layers: int
    num_classes: int

class EyeblinkArchitectureConfig(BaseModel):
    base_model_name: str
    lstm_hidden_size: int
    dropout_rate: float
    img_size: Tuple[int, int]

# New architecture config for V4 to include dropout
class SiglipArchitectureV4Config(SiglipArchitectureConfig):
    dropout_rate: float = 0.5

# Base model for shared configuration fields
class BaseModelConfig(BaseModel):
    class_name: str
    description: str
    model_path: str
    device: str = "cuda"
    isDetailed: bool = False
    # FIX: Added isAudio and isVideo flags with defaults to ensure all model configs are validated.
    # This makes the system more robust against misconfiguration.
    isAudio: bool = False
    isVideo: bool = True

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
    
class EyeblinkModelConfig(BaseModelConfig):
    class_name: Literal["EYEBLINK-CNN-LSTM-V1"]
    # FIX: Added the missing dlib_model_path to match the YAML file.
    dlib_model_path: str
    sequence_length: int
    blink_threshold: float
    consecutive_frames: int
    model_definition: EyeblinkArchitectureConfig

class ScatteringWaveV1Config(BaseModelConfig):
    class_name: Literal["SCATTERING-WAVE-V1"]
    sampling_rate: int
    duration_seconds: float
    image_size: Tuple[int, int]

# A Discriminated Union to allow only the available models to be served
ModelConfig = Annotated[
    Union[
        ColorCuesConfig,
        SiglipLSTMv1Config,
        SiglipLSTMv3Config,
        SiglipLSTMv4Config,
        EfficientNetB7Config,
        EyeblinkModelConfig,
        ScatteringWaveV1Config
    ],
    Field(discriminator="class_name"),
]

# The main Settings class that loads from both .env and config.yaml
class Settings(BaseSettings):
    api_key: SecretStr
    default_model_name: str
    project_name: str
    device: str
    active_models: str
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
        
        try:
            from dotenv import load_dotenv
            load_dotenv()
            logger.info("🔧 Loaded environment variables from .env file.")
        except ImportError:
            logger.warning("⚠️ python-dotenv not found, using system environment variables only.")
        
        env_device = os.getenv('DEVICE', 'cpu').lower()
        logger.info(f"🔧 [High Priority] Device set to '{env_device}' from environment variable.")
        
        try:
            with open(yaml_path, 'r') as file:
                yaml_config = yaml.safe_load(file)
                if not yaml_config:
                    raise ValueError("YAML file is empty or invalid.")
        except FileNotFoundError:
            raise RuntimeError(f"Configuration file '{yaml_path}' not found.")
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing YAML file '{yaml_path}': {e}")

        # Override global and all model-specific device settings with the environment variable.
        # This ensures the environment setting is the single source of truth for the compute device.
        yaml_config['device'] = env_device
        if yaml_config.get('models'):
            for model_name in yaml_config['models']:
                if isinstance(yaml_config['models'][model_name], dict):
                    yaml_config['models'][model_name]['device'] = env_device

        # Temporarily load settings to get the ACTIVE_MODELS list from .env
        temp_settings = cls(**yaml_config)
        
        active_model_list = temp_settings.active_model_list
        all_configured_models = yaml_config.get('models', {})
        
        # Filter the models from YAML to only include those specified in ACTIVE_MODELS
        filtered_models = {
            name: config for name, config in all_configured_models.items() if name in active_model_list
        }
        
        if len(filtered_models) != len(active_model_list):
            logger.warning(
                f"⚠️ Mismatch between ACTIVE_MODELS and config.yaml. "
                f"Requested: {active_model_list}. Found in YAML: {list(all_configured_models.keys())}"
            )

        yaml_config['models'] = filtered_models
        
        if filtered_models:
            logger.info(f"🔧 Loading configurations for active models: {', '.join(filtered_models.keys())}")
        else:
            logger.warning("⚠️ No active models are configured to be loaded!")

        return cls(**yaml_config)

try:
    settings = Settings.from_yaml("configs/config.yaml")
except (RuntimeError, ValueError, ValidationError) as e:
    logger.critical(f"❌ FATAL: Could not load configuration. Server cannot start. Error: {e}")
    # Use sys.exit(1) for a more forceful shutdown on critical config errors.
    import sys
    sys.exit(1)