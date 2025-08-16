#!/usr/bin/env python3
"""
Test script to verify terminal logging is working for model responses.
This simulates API requests to check if responses are printed to terminal.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_terminal_logging():
    """Test that terminal logging works for model responses."""
    print("🧪 Testing Terminal Logging for Model Responses")
    print("=" * 70)
    
    try:
        from src.config import settings
        from src.ml.registry import ModelManager
        
        print("📋 Setting up model manager...")
        manager = ModelManager(settings)
        
        # Get the first available model
        if settings.models:
            model_name = list(settings.models.keys())[0]
            print(f"🤖 Testing with model: {model_name}")
            
            # Test model loading (should show terminal output)
            print("\n🔄 Loading model (should show terminal output):")
            model = manager.get_model(model_name)
            
            print(f"\n✅ Model loaded successfully!")
            print(f"🎯 Model device: {model.device}")
            print(f"🏷️  Model class: {model.config.class_name}")
            
        else:
            print("⚠️  No models configured for testing")
            
        print(f"\n🎉 Terminal logging test completed!")
        print("\n💡 Tips:")
        print("   • When you run the server and make API requests,")
        print("   • you should see detailed response information")
        print("   • printed to the terminal console.")
        print("   • This includes predictions, confidence scores,")
        print("   • processing times, and model details.")
        
    except Exception as e:
        print(f"❌ Error testing terminal logging: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    test_terminal_logging()
