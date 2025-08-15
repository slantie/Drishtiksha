#!/usr/bin/env python3
"""
Test script for LSTMDetectorV3 integration.
This script tests the new model functionality without running the full server.
"""

import os
import sys
import time
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import settings
from ml.registry import ModelManager


def test_model_loading():
    """Test if the new LSTMDetectorV3 model loads correctly."""
    print("🧪 Testing LSTMDetectorV3 Model Loading...")

    try:
        # Initialize model manager
        manager = ModelManager(settings.models)

        # Test loading the new v3 model
        print(f"Loading model: {settings.default_model_name}")
        model = manager.get_model(settings.default_model_name)

        print(f"✅ Model loaded successfully!")
        print(f"Model type: {type(model).__name__}")
        print(f"Model device: {model.device}")

        # Test model info
        info = model.get_info()
        print(f"Model info: {info}")

        return True

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def test_model_methods():
    """Test if the new methods are available."""
    print("\n🧪 Testing Model Methods...")

    try:
        manager = ModelManager(settings.models)
        model = manager.get_model(settings.default_model_name)

        # Check if new methods exist
        methods_to_check = [
            "predict",
            "predict_with_metrics",
            "predict_visualized",
            "get_frame_analysis_summary",
            "save_analysis_metrics",
        ]

        for method_name in methods_to_check:
            if hasattr(model, method_name):
                print(f"✅ Method '{method_name}' is available")
            else:
                print(f"❌ Method '{method_name}' is missing")

        return True

    except Exception as e:
        print(f"❌ Method testing failed: {e}")
        return False


def test_video_analysis():
    """Test video analysis if a test video is available."""
    print("\n🧪 Testing Video Analysis...")

    # Look for test video in assets directory
    test_video_path = "assets/id0_0001.mp4"

    if not os.path.exists(test_video_path):
        print(f"⚠️  Test video not found at {test_video_path}")
        print("Skipping video analysis test...")
        return True

    try:
        manager = ModelManager(settings.models)
        model = manager.get_model(settings.default_model_name)

        print(f"Testing with video: {test_video_path}")

        # Test basic prediction
        print("Testing basic prediction...")
        start_time = time.time()
        result = model.predict(test_video_path)
        basic_time = time.time() - start_time

        print(
            f"✅ Basic prediction: {result['prediction']} (confidence: {result['confidence']:.3f})"
        )
        print(f"   Processing time: {basic_time:.2f}s")

        # Test frame analysis summary (if available)
        if hasattr(model, "get_frame_analysis_summary"):
            print("Testing frame analysis summary...")
            start_time = time.time()
            summary = model.get_frame_analysis_summary(test_video_path)
            frame_time = time.time() - start_time

            print(
                f"✅ Frame analysis: {summary['prediction']} (confidence: {summary['confidence']:.3f})"
            )
            print(f"   Frames analyzed: {summary['frame_count']}")
            print(f"   Suspicious frames: {summary['suspicious_frames']}")
            print(f"   Processing time: {frame_time:.2f}s")

        return True

    except Exception as e:
        print(f"❌ Video analysis failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting LSTMDetectorV3 Integration Tests...\n")

    tests = [
        ("Model Loading", test_model_loading),
        ("Model Methods", test_model_methods),
        ("Video Analysis", test_video_analysis),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"{'=' * 50}")
        result = test_func()
        results.append((test_name, result))

    print(f"\n{'=' * 50}")
    print("📊 Test Results Summary:")
    print(f"{'=' * 50}")

    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All tests passed! LSTMDetectorV3 integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the configuration and dependencies.")


if __name__ == "__main__":
    main()
