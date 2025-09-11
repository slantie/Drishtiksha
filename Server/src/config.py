# src/config.py

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Tuple, Union, Annotated, Literal, List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import (
    BaseModel, Field, SecretStr, ValidationError, model_validator, AfterValidator
)

logger = logging.getLogger(__name__)

def check_path_exists(path: Path) -> Path:
    if not path.exists():
        raise ValueError(f"Configuration error: File path does not exist: '{path}'")
    if not path.is_file():
        raise ValueError(f"Configuration error: Path is not a file: '{path}'")
    return path

ExistingPath = Annotated[Path, AfterValidator(check_path_exists)]


# --- Model Architecture Schemas (Unchanged) ---
class SiglipArchitectureConfig(BaseModel):
    base_model_path: str
    lstm_hidden_size: int
    lstm_num_layers: int
    num_classes: int

class SiglipArchitectureV4Config(SiglipArchitectureConfig):
    dropout_rate: float = 0.5

class EyeblinkArchitectureConfig(BaseModel):
    base_model_name: str
    lstm_hidden_size: int
    dropout_rate: float
    img_size: Tuple[int, int]


# --- Main Model Configuration Schemas ---

class BaseModelConfig(BaseModel):
    class_name: str
    description: str
    model_path: ExistingPath
    device: str = "cuda"
    isAudio: bool = False
    isVideo: bool = True
    isImage: bool = False

class SiglipLSTMv1Config(BaseModelConfig):
    # FIX: The Literal must match the Python class name from config.yaml
    class_name: Literal["SiglipLSTMV1"]
    processor_path: str
    num_frames: int
    model_definition: SiglipArchitectureConfig

class SiglipLSTMv3Config(BaseModelConfig):
    # FIX: The Literal must match the Python class name from config.yaml
    class_name: Literal["SiglipLSTMV3"]
    processor_path: str
    num_frames: int
    rolling_window_size: int
    model_definition: SiglipArchitectureConfig

class SiglipLSTMv4Config(BaseModelConfig):
    # FIX: The Literal must match the Python class name from config.yaml
    class_name: Literal["SiglipLSTMV4"]
    processor_path: str
    num_frames: int
    rolling_window_size: int
    model_definition: SiglipArchitectureV4Config

class ColorCuesConfig(BaseModelConfig):
    # FIX: The Literal must match the Python class name from config.yaml
    class_name: Literal["ColorCuesLSTMV1"]
    dlib_model_path: ExistingPath
    sequence_length: int
    frames_per_video: int
    histogram_bins: int
    landmark_margin: int
    rolling_window_size: int
    hidden_size: int
    dropout: float

class EfficientNetB7Config(BaseModelConfig):
    # FIX: The Literal must match the Python class name from config.yaml
    class_name: Literal["EfficientNetB7Detector"]
    encoder: str
    input_size: int

class EyeblinkModelConfig(BaseModelConfig):
    # FIX: The Literal must match the Python class name from config.yaml
    class_name: Literal["EyeblinkDetectorV1"]
    dlib_model_path: ExistingPath
    sequence_length: int
    blink_threshold: float
    consecutive_frames: int
    model_definition: EyeblinkArchitectureConfig

class ScatteringWaveV1Config(BaseModelConfig):
    # FIX: The Literal must match the Python class name from config.yaml
    class_name: Literal["ScatteringWaveV1"]
    sampling_rate: int
    duration_seconds: float
    image_size: Tuple[int, int]


# A Discriminated Union to validate and parse the correct model config.
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

# --- Main Application Settings Class (Unchanged) ---
class Settings(BaseSettings):
    api_key: SecretStr
    default_model_name: str
    active_models: str
    device: str
    project_name: str
    models: Dict[str, ModelConfig]

    assets_base_url: str = Field(..., alias="ASSETS_BASE_URL")
    storage_path: Path = Field(..., alias="STORAGE_PATH")
    
    redis_url: Optional[str] = Field(None, alias="REDIS_URL")
    media_progress_channel_name: str = Field("media-progress-events", alias="MEDIA_PROGRESS_CHANNEL_NAME")
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

    @property
    def active_model_list(self) -> List[str]:
        if not self.active_models: return []
        return [model.strip() for model in self.active_models.split(',') if model.strip()]
        
    @property
    def active_model_list(self) -> List[str]:
        if not self.active_models: return []
        return [model.strip() for model in self.active_models.split(',') if model.strip()]
        
    @model_validator(mode='after')
    def validate_paths_and_models(self) -> 'Settings':
        # Validate default model
        active_list = self.active_model_list
        if active_list and self.default_model_name not in active_list:
            raise ValueError(
                f"Config error: Default model '{self.default_model_name}' not in ACTIVE_MODELS."
            )
        
        # Validate and create storage path
        if not self.storage_path.exists():
            logger.info(f"Storage path '{self.storage_path}' not found. Creating it.")
            self.storage_path.mkdir(parents=True, exist_ok=True)
        elif not self.storage_path.is_dir():
            raise ValueError(f"Config error: STORAGE_PATH '{self.storage_path}' must be a directory.")
        
        return self

    @classmethod
    def from_yaml_and_env(cls, yaml_path: str, env_file: str = '.env') -> "Settings":
        from dotenv import load_dotenv
        load_dotenv(env_file)
        logger.info(f"🔧 Loaded environment variables from '{env_file}'.")

        env_device = os.getenv('DEVICE', 'cpu').lower()
        logger.info(f"🔧 Device set to '{env_device}' from environment.")

        try:
            with open(yaml_path, 'r') as file:
                yaml_data = yaml.safe_load(file) or {}
        except FileNotFoundError:
            raise RuntimeError(f"FATAL: Config file '{yaml_path}' not found.")
        except yaml.YAMLError as e:
            raise RuntimeError(f"FATAL: Error parsing YAML file '{yaml_path}': {e}")
        
        env_data = {
            "api_key": os.getenv("API_KEY"),
            "default_model_name": os.getenv("DEFAULT_MODEL_NAME"),
            "active_models": os.getenv("ACTIVE_MODELS"),
            "device": env_device,
            "ASSETS_BASE_URL": os.getenv("ASSETS_BASE_URL"),
            "STORAGE_PATH": os.getenv("STORAGE_PATH", "../Backend/public/media"),
        }
        
        merged_config = {**yaml_data, **{k: v for k, v in env_data.items() if v is not None}}
        
        if merged_config.get('models'):
            for model_name in merged_config['models']:
                if isinstance(merged_config['models'][model_name], dict):
                    merged_config['models'][model_name]['device'] = env_device

        active_model_names = [m.strip() for m in (env_data.get("active_models") or "").split(',') if m.strip()]
        
        if 'models' in merged_config:
            merged_config['models'] = {
                name: config for name, config in merged_config['models'].items() if name in active_model_names
            }

        return cls(**merged_config)

try:
    settings = Settings.from_yaml_and_env("configs/config.yaml")
except (RuntimeError, ValueError, ValidationError) as e:
    logger.critical(f"❌ FATAL: Could not load configuration. Server cannot start.\nError: {e}")
    import sys
    sys.exit(1)