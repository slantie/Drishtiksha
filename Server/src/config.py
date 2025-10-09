# src/config.py

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Tuple, Union, Annotated, Literal, List, Optional, Any

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import (
    BaseModel, Field, SecretStr, ValidationError, model_validator, AfterValidator
)

logger = logging.getLogger(__name__)

def check_model_dir_exists(path: Path) -> Path:
    logger.info(f"MFF-MoE-v1 Models Folder Path: {path}")
    if not path.is_dir():
        raise ValueError(f"Configuration error: Path is not a directory: '{path}'")
    weight_file = path / "MFF-MoE-v1.pth"
    state_file = path / "MFF-MoE-v1.state"
    if not weight_file.exists() or not state_file.exists():
        raise ValueError(f"Configuration error: Directory '{path}' must contain 'MFF-MoE-v1.pth' and 'MFF-MoE-v1.state'")
    return path

ExistingModelDir = Annotated[Path, AfterValidator(check_model_dir_exists)]

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

class CrossEfficientViTArchConfig(BaseModel):
    image_size: int = Field(..., alias='image-size')
    num_classes: int = Field(..., alias='num-classes')
    depth: int
    sm_dim: int = Field(..., alias='sm-dim')
    sm_patch_size: int = Field(..., alias='sm-patch-size')
    sm_enc_depth: int = Field(..., alias='sm-enc-depth')
    sm_enc_dim_head: int = Field(..., alias='sm-enc-dim-head')
    sm_enc_heads: int = Field(..., alias='sm-enc-heads')
    sm_enc_mlp_dim: int = Field(..., alias='sm-enc-mlp-dim')
    lg_dim: int = Field(..., alias='lg-dim')
    lg_patch_size: int = Field(..., alias='lg-patch-size')
    lg_enc_depth: int = Field(..., alias='lg-enc-depth')
    lg_enc_dim_head: int = Field(..., alias='lg-enc-dim-head')
    lg_enc_heads: int = Field(..., alias='lg-enc-heads')
    lg_enc_mlp_dim: int = Field(..., alias='lg-enc-mlp-dim')
    cross_attn_depth: int = Field(..., alias='cross-attn-depth')
    cross_attn_dim_head: int = Field(..., alias='cross-attn-dim-head')
    cross_attn_heads: int = Field(..., alias='cross-attn-heads')
    lg_channels: int = Field(..., alias='lg-channels')
    sm_channels: int = Field(..., alias='sm-channels')
    dropout: float
    emb_dropout: float = Field(..., alias='emb-dropout')

class LipFDv1ArchConfig(BaseModel):
    arch: str
    n_extract: int
    window_len: int

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

class MelSpectrogramCNNConfig(BaseModelConfig):
    class_name: Literal["MelSpectrogramCNNV1"]
    sampling_rate: int
    n_fft: int
    hop_length: int
    n_mels: int
    dpi: int
    chunk_duration_s: float

class STFTSpectrogramCNNConfig(BaseModelConfig):
    class_name: Literal["STFTSpectrogramCNNV1"]
    sampling_rate: int
    n_fft_wide: int
    n_fft_narrow: int
    hop_length: int
    img_height: int
    img_width: int
    dpi: int
    chunk_duration_s: float

class DistilDIREv1Config(BaseModelConfig):
    class_name: Literal["DistilDIREDetectorV1"]
    adm_model_path: ExistingPath
    image_size: int = 256
    adm_config: Dict[str, Any] = Field(default_factory=dict)
    isImage: bool = True
    isVideo: bool = False
    
class MFFMoEV1Config(BaseModelConfig):
    class_name: Literal["MFFMoEDetectorV1"]
    model_path: ExistingModelDir
    video_frames_to_sample: int = 100
    isImage: bool = True
    isVideo: bool = True
    isAudio: bool = False

class CrossEfficientViTConfig(BaseModelConfig):
    class_name: Literal["CrossEfficientViTDetector"]
    model_definition: CrossEfficientViTArchConfig

class LipFDv1Config(BaseModelConfig):
    class_name: Literal["LipFDetectorV1"]
    model_definition: LipFDv1ArchConfig

# A Discriminated Union to validate and parse the correct model config.
ModelConfig = Annotated[
    Union[
        ColorCuesConfig,
        SiglipLSTMv1Config,
        SiglipLSTMv3Config,
        SiglipLSTMv4Config,
        EfficientNetB7Config,
        EyeblinkModelConfig,
        ScatteringWaveV1Config,
        MelSpectrogramCNNConfig,
        STFTSpectrogramCNNConfig,
        DistilDIREv1Config,
        MFFMoEV1Config,
        CrossEfficientViTConfig,
        LipFDv1Config
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

        default_model_name_env = os.getenv("DEFAULT_MODEL_NAME")
        
        env_data = {
            "api_key": os.getenv("API_KEY"),
            "default_model_name": default_model_name_env.strip() if default_model_name_env else None,
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

    @classmethod
    def from_yaml_and_env_cli(cls, yaml_path: str, env_file: str = '.env') -> "Settings":
        """
        Load settings for CLI mode - includes ALL models from config.yaml,
        not just ACTIVE_MODELS. This allows CLI to access all configured models.
        """
        from dotenv import load_dotenv
        load_dotenv(env_file)
        logger.info(f"🔧 [CLI MODE] Loaded environment variables from '{env_file}'.")

        env_device = os.getenv('DEVICE', 'cpu').lower()
        logger.info(f"🔧 [CLI MODE] Device set to '{env_device}' from environment.")

        try:
            with open(yaml_path, 'r') as file:
                yaml_data = yaml.safe_load(file) or {}
        except FileNotFoundError:
            raise RuntimeError(f"FATAL: Config file '{yaml_path}' not found.")
        except yaml.YAMLError as e:
            raise RuntimeError(f"FATAL: Error parsing YAML file '{yaml_path}': {e}")

        # For CLI, use first model as default if not specified
        default_model_name_env = os.getenv("DEFAULT_MODEL_NAME")
        if not default_model_name_env and yaml_data.get('models'):
            default_model_name_env = list(yaml_data['models'].keys())[0]
            logger.info(f"🔧 [CLI MODE] No DEFAULT_MODEL_NAME set, using first model: {default_model_name_env}")
        
        env_data = {
            "api_key": os.getenv("API_KEY", "cli-mode-key"),  # CLI doesn't need API key
            "default_model_name": default_model_name_env.strip() if default_model_name_env else None,
            "active_models": ",".join(yaml_data.get('models', {}).keys()),  # ALL models
            "device": env_device,
            "ASSETS_BASE_URL": os.getenv("ASSETS_BASE_URL", "http://localhost:3000"),
            "STORAGE_PATH": os.getenv("STORAGE_PATH", "./storage"),
        }
        
        merged_config = {**yaml_data, **{k: v for k, v in env_data.items() if v is not None}}
        
        # Set device for all models
        if merged_config.get('models'):
            for model_name in merged_config['models']:
                if isinstance(merged_config['models'][model_name], dict):
                    merged_config['models'][model_name]['device'] = env_device

        # CLI MODE: Include ALL models, not just active ones
        logger.info(f"🔧 [CLI MODE] Loaded {len(merged_config.get('models', {}))} models from config")

        return cls(**merged_config)

try:
    settings = Settings.from_yaml_and_env("configs/config.yaml")
except (RuntimeError, ValueError, ValidationError) as e:
    logger.critical(f"❌ FATAL: Could not load configuration. Server cannot start.\nError: {e}")
    import sys
    sys.exit(1)