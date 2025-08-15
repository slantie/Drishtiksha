#!/usr/bin/env python3
"""
Test script for ColorCues LSTM detector integration.
Tests the ColorCues model endpoints and functionality.
"""

import requests
import os
import sys

API_URL = "http://localhost:8000"
API_KEY = "98e75012eea105151f4ddbab76872dd3717c74247903b302f74abdd783d0c5bb"
VIDEO_PATH = "assets/id0_0001.mp4"


def test_analyze_colorcues():
    """Test the /analyze endpoint with ColorCues model."""
    print("Testing /analyze with ColorCues model...")

    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Test video not found at {VIDEO_PATH}")
        return False

    try:
        with open(VIDEO_PATH, "rb") as f:
            files = {"video": (VIDEO_PATH, f, "video/mp4")}
            data = {
                "video_id": "test-colorcues-analyze",
                "model_name": "color-cues-lstm-v1",  # Specify ColorCues model
            }
            headers = {"X-API-Key": API_KEY}

            resp = requests.post(
                f"{API_URL}/analyze", files=files, data=data, headers=headers
            )

        if resp.status_code == 200:
            result = resp.json()
            print("✅ ColorCues /analyze endpoint working")
            print(f"Response: {result}")
            return True
        else:
            print(f"❌ ColorCues /analyze failed with status {resp.status_code}")
            print(f"Error: {resp.text}")
            return False

    except Exception as e:
        print(f"❌ ColorCues /analyze error: {e}")
        return False


def test_analyze_colorcues_detailed():
    """Test the /analyze/detailed endpoint with ColorCues model."""
    print("Testing /analyze/detailed with ColorCues model...")

    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Test video not found at {VIDEO_PATH}")
        return False

    try:
        with open(VIDEO_PATH, "rb") as f:
            files = {"video": (VIDEO_PATH, f, "video/mp4")}
            data = {
                "video_id": "test-colorcues-detailed",
                "model_name": "color-cues-lstm-v1",  # Specify ColorCues model
            }
            headers = {"X-API-Key": API_KEY}

            resp = requests.post(
                f"{API_URL}/analyze/detailed", files=files, data=data, headers=headers
            )

        if resp.status_code == 200:
            result = resp.json()
            print("✅ ColorCues /analyze/detailed endpoint working")
            print(f"Response: {result}")
            return True
        else:
            print(
                f"❌ ColorCues /analyze/detailed failed with status {resp.status_code}"
            )
            print(f"Error: {resp.text}")
            return False

    except Exception as e:
        print(f"❌ ColorCues /analyze/detailed error: {e}")
        return False


def test_model_info_colorcues():
    """Test model info for ColorCues model."""
    print("Testing model info for ColorCues...")

    try:
        headers = {"X-API-Key": API_KEY}
        resp = requests.get(f"{API_URL}/model/info", headers=headers)

        if resp.status_code == 200:
            result = resp.json()
            print("✅ Model info endpoint working")
            print(f"Current model info: {result}")
            return True
        else:
            print(f"❌ Model info failed with status {resp.status_code}")
            return False

    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False


def main():
    """Run ColorCues integration tests."""
    print("🧪 Starting ColorCues LSTM integration tests...")
    print("=" * 60)

    if len(sys.argv) > 1:
        global VIDEO_PATH
        VIDEO_PATH = sys.argv[1]
        print(f"Using video: {VIDEO_PATH}")

    all_passed = True

    # Test model info
    if not test_model_info_colorcues():
        all_passed = False

    print("-" * 40)

    # Test basic analysis
    if not test_analyze_colorcues():
        all_passed = False

    print("-" * 40)

    # Test detailed analysis (if ColorCues supports it)
    if not test_analyze_colorcues_detailed():
        print("⚠️  Detailed analysis may not be implemented for ColorCues yet")

    print("=" * 60)

    if all_passed:
        print("🎉 ColorCues integration tests completed successfully!")
        print("\nNext steps:")
        print("1. Test with different video files")
        print("2. Compare results with SIGLIP-LSTM model")
        print("3. Verify ColorCues model files are properly placed")
        sys.exit(0)
    else:
        print("💥 Some ColorCues tests failed.")
        print("\nTroubleshooting:")
        print("1. Ensure ColorCues model files exist:")
        print("   - models/best_color_cues_lstm_model.pth")
        print("   - models/shape_predictor_68_face_landmarks.dat")
        print("2. Check server logs for detailed error messages")
        print("3. Verify dlib is properly installed")
        sys.exit(1)


if __name__ == "__main__":
    main()
