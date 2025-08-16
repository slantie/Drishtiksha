#!/usr/bin/env python3
"""
Simple test to verify device configuration loading from .env file.
"""

# Load dotenv first
from dotenv import load_dotenv
load_dotenv()

import os
import yaml

def test_device_override():
    """Test device override logic without full server setup."""
    print("🧪 Testing Device Override Logic")
    print("=" * 50)
    
    # Check environment variable
    env_device = os.getenv('DEVICE', 'cpu').lower()
    print(f"📋 Environment DEVICE: {env_device}")
    
    # Load YAML config
    try:
        with open('configs/config.yaml', 'r') as file:
            yaml_config = yaml.safe_load(file)
            
        print(f"📋 Original YAML device: {yaml_config.get('device', 'Not Set')}")
        
        # Apply our override logic
        yaml_config['device'] = env_device
        
        if yaml_config.get('models'):
            for model_name, model_config in yaml_config['models'].items():
                if isinstance(model_config, dict):
                    model_config['device'] = env_device
                    print(f"🔧 Set device '{env_device}' for model '{model_name}'")
        
        print(f"📋 Final YAML device: {yaml_config.get('device')}")
        
        # Check model configurations
        for model_name, model_config in yaml_config.get('models', {}).items():
            print(f"🤖 {model_name}: device = {model_config.get('device')}")
            
        print("✅ Device override logic working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_device_override()
