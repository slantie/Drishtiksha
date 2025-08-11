# /home/dell-pc-03/Deepfake/deepfake-detection/Raj/src/utils.py

import yaml

def load_config(config_path):
    """Loads a YAML configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config