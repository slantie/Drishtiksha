# src/cli/config.py

"""CLI-specific configuration loader that includes ALL models."""

from src.config import Settings

# Global CLI settings instance
_cli_settings = None


def get_cli_settings() -> Settings:
    """
    Get CLI settings with ALL models loaded (not just ACTIVE_MODELS).
    This is cached after first load.
    """
    global _cli_settings
    
    if _cli_settings is None:
        _cli_settings = Settings.from_yaml_and_env_cli("configs/config.yaml")
    
    return _cli_settings
