#!/usr/bin/env python3
"""
Test script to verify all models have the required methods for complete analysis
"""

import sys
import os

sys.path.append("src")


def test_model_methods():
    """Test that all models have the required methods"""

    # Import the model classes
    try:
        from src.ml.models.lstm_detector_v2 import LSTMDetector  # SIGLIP_LSTM_V1
        from src.ml.models.lstm_detector_v3 import LSTMDetectorV3  # SIGLIP_LSTM_V3
        from src.ml.models.color_cues_detector import (
            ColorCuesDetector,
        )  # COLOR_CUES_LSTM_V1

        print("✅ All model imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    # Required methods for complete functionality
    required_methods = [
        "predict",  # Basic analysis
        "predict_with_metrics",  # Detailed analysis
        "get_frame_analysis_summary",  # Frame analysis
        "predict_visualized",  # Visualization
    ]

    models = {
        "SIGLIP_LSTM_V1": LSTMDetector,
        "SIGLIP_LSTM_V3": LSTMDetectorV3,
        "COLOR_CUES_LSTM_V1": ColorCuesDetector,
    }

    print("\nTesting model methods:")
    all_good = True

    for model_name, model_class in models.items():
        print(f"\n{model_name}:")
        for method in required_methods:
            if hasattr(model_class, method):
                print(f"  ✅ {method}")
            else:
                print(f"  ❌ {method} - MISSING")
                all_good = False

    if all_good:
        print("\n🎉 All models have all required methods!")
        return True
    else:
        print("\n💥 Some models are missing required methods")
        return False


if __name__ == "__main__":
    test_model_methods()
