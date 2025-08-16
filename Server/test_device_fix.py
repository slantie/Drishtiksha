#!/usr/bin/env python3
"""
Test script to verify device configuration is working correctly.
This will check that models load on the device specified in the environment variable.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_device_configuration():
    """Test that device configuration is respected."""
    print("🧪 Testing Device Configuration Fix")
    print("=" * 60)
    
    try:
        from src.config import settings
        from src.ml.registry import ModelManager
        from src.ml.system_info import get_device_info, get_server_configuration
        
        print("📋 Configuration Summary:")
        print("=" * 30)
        
        # Show environment device setting
        print(f"🔧 Environment Device Setting: {settings.device}")
        
        # Show device info
        device_info = get_device_info()
        print(f"🖥️  Detected Device Type: {device_info.type}")
        print(f"🏷️  Device Name: {device_info.name}")
        
        # Show model configurations
        print(f"\n📊 Model Configurations:")
        print("=" * 30)
        for model_name, model_config in settings.models.items():
            print(f"📦 {model_name}: Device = {model_config.device}")
        
        # Test model loading
        print(f"\n🧠 Testing Model Loading:")
        print("=" * 30)
        manager = ModelManager(settings)
        
        # Try to load one model to verify device setting
        if settings.models:
            first_model_name = list(settings.models.keys())[0]
            print(f"🔄 Loading model: {first_model_name}")
            model = manager.get_model(first_model_name)
            print(f"✅ Model loaded successfully")
            print(f"🎯 Model device: {model.device}")
            
            # Check if the model's device matches environment setting
            if model.device.lower() == settings.device.lower():
                print("✅ Device configuration is working correctly!")
            else:
                print(f"❌ Device mismatch! Expected: {settings.device}, Got: {model.device}")
        else:
            print("⚠️  No models configured for testing")
            
        print(f"\n🎉 Device configuration test completed!")
        
    except Exception as e:
        print(f"❌ Error testing device configuration: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    test_device_configuration()
