# src/config.py

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, Any

class AppSettings(BaseSettings):
    """Manages application settings loaded from .env and config.yaml."""
    # From .env
    api_key: str
    default_model_name: str

    # From YAML
    project_name: str
    device: str
    models: Dict[str, Any]

    # The CHANGE is here: we add extra='ignore' to the config.
    # This tells Pydantic to simply ignore fields from the input
    # that are not defined in this model (like the 'training' block).
    model_config = SettingsConfigDict(
        env_file='.env', 
        env_file_encoding='utf-8', 
        extra='ignore'  # <-- ADD THIS LINE
    )

def load_settings() -> AppSettings:
    """Loads settings from both YAML and .env file."""
    try:
        with open('configs/config.yaml', 'r') as file:
            yaml_config = yaml.safe_load(file)
    except FileNotFoundError:
        raise RuntimeError("Configuration file 'configs/config.yaml' not found.")

    # Pydantic will now correctly initialize AppSettings using the fields it knows
    # from the YAML and the .env file, and it will ignore the 'training' key.
    return AppSettings(**yaml_config)

settings = load_settings()