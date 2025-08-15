# src/config.py

import yaml
from pydantic import BaseModel, Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, Union, Annotated, Literal

# A dedicated Pydantic model for the SigLIP architecture details
class SiglipArchitectureConfig(BaseModel):
    base_model_path: str
    lstm_hidden_size: int
    lstm_num_layers: int
    num_classes: int

# Base model for shared configuration fields
class BaseModelConfig(BaseModel):
    class_name: str
    description: str
    model_path: str
    device: str

# Specific configuration model for siglip-lstm-v1
class SiglipLSTMv1Config(BaseModelConfig):
    class_name: Literal["LSTMDetector"]
    processor_path: str
    num_frames: int
    model_definition: SiglipArchitectureConfig # Nested architecture details

# Specific configuration model for siglip-lstm-v3
class SiglipLSTMv3Config(BaseModelConfig):
    class_name: Literal["LSTMDetectorV3"]
    processor_path: str
    num_frames: int
    rolling_window_size: int
    model_definition: SiglipArchitectureConfig # Nested architecture details

# Specific configuration model for color-cues-lstm-v1
class ColorCuesConfig(BaseModelConfig):
    class_name: Literal["ColorCuesDetector"]
    dlib_model_path: str
    sequence_length: int
    frames_per_video: int
    histogram_bins: int
    landmark_margin: int
    rolling_window_size: int
    hidden_size: int
    dropout: float

# A Discriminated Union to allow Pydantic to parse the correct model config
ModelConfig = Annotated[
    Union[SiglipLSTMv1Config, SiglipLSTMv3Config, ColorCuesConfig],
    Field(discriminator="class_name"),
]

# The main Settings class that loads from both .env and config.yaml
class Settings(BaseSettings):
    api_key: SecretStr
    default_model_name: str
    project_name: str
    device: str
    models: Dict[str, ModelConfig]

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'  # Ignore extra fields like 'training' from the YAML
    )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Settings":
        """Factory method to load settings from a YAML file and merge with .env."""
        try:
            with open(yaml_path, 'r') as file:
                yaml_config = yaml.safe_load(file)
                if not yaml_config:
                    raise ValueError("YAML file is empty or invalid.")
        except FileNotFoundError:
            raise RuntimeError(f"Configuration file '{yaml_path}' not found.")
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing YAML file '{yaml_path}': {e}")

        return cls(**yaml_config)

# Singleton instance of the settings, used throughout the application
try:
    settings = Settings.from_yaml("configs/config.yaml")
except (RuntimeError, ValueError, ValidationError) as e:
    print(f"FATAL: Could not load configuration. {e}")
    exit(1)