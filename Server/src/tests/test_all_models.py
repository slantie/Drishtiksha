#!/usr/bin/env python3
"""
Comprehensive test script for all models including ColorCues.
Tests all endpoints with multiple models for comparison.
"""

import requests
import time
import sys
import os

API_URL = "http://localhost:8000"
API_KEY = "98e75012eea105151f4ddbab76872dd3717c74247903b302f74abdd783d0c5bb"
VIDEO_PATH = "assets/id0_0001.mp4"

# Available models to test
MODELS_TO_TEST = ["siglip-lstm-v3", "color-cues-lstm-v1"]


def test_ping():
    """Test the ping endpoint."""
    try:
        response = requests.get(f"{API_URL}/ping", timeout=10)
        if response.status_code == 200:
            print("✅ Server is responsive")
            return True
        else:
            print(f"❌ Ping failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ping error: {e}")
        return False


def test_health():
    """Test the health endpoint."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=15)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed - Model: {data.get('default_model')}")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def test_analyze_with_model(model_name: str):
    """Test the /analyze endpoint with a specific model."""
    print(f"Testing /analyze with {model_name}...")

    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Test video not found at {VIDEO_PATH}")
        return False

    try:
        start_time = time.time()

        with open(VIDEO_PATH, "rb") as f:
            files = {"video": (VIDEO_PATH, f, "video/mp4")}
            data = {"video_id": f"test-{model_name}", "model_name": model_name}
            headers = {"X-API-Key": API_KEY}

            response = requests.post(
                f"{API_URL}/analyze", files=files, data=data, headers=headers
            )

        request_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            analysis_result = result.get("result", {})

            print(f"✅ {model_name} analysis completed")
            print(f"   Prediction: {analysis_result.get('prediction')}")
            print(f"   Confidence: {analysis_result.get('confidence', 0):.3f}")
            print(
                f"   Processing Time: {analysis_result.get('processing_time', 0):.2f}s"
            )
            print(f"   Request Time: {request_time:.2f}s")

            # Model-specific info
            if "fake_probability" in analysis_result:
                print(
                    f"   Fake Probability: {analysis_result.get('fake_probability'):.3f}"
                )
            if "num_sequences" in analysis_result:
                print(f"   Sequences Analyzed: {analysis_result.get('num_sequences')}")

            return True
        else:
            print(f"❌ {model_name} analysis failed with status {response.status_code}")
            print(f"   Error: {response.text}")
            return False

    except Exception as e:
        print(f"❌ {model_name} analysis error: {e}")
        return False


def test_detailed_analysis_with_model(model_name: str):
    """Test the /analyze/detailed endpoint with a specific model."""
    print(f"Testing /analyze/detailed with {model_name}...")

    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Test video not found at {VIDEO_PATH}")
        return False

    try:
        with open(VIDEO_PATH, "rb") as f:
            files = {"video": (VIDEO_PATH, f, "video/mp4")}
            data = {"video_id": f"test-detailed-{model_name}", "model_name": model_name}
            headers = {"X-API-Key": API_KEY}

            response = requests.post(
                f"{API_URL}/analyze/detailed", files=files, data=data, headers=headers
            )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ {model_name} detailed analysis completed")

            # Check for detailed metrics
            analysis_result = result.get("result", {})
            if "metrics" in analysis_result:
                metrics = analysis_result["metrics"]
                print(f"   Frame Count: {metrics.get('frame_count', 0)}")
                print(
                    f"   Suspicious Frames: {metrics.get('suspicious_frames_count', 0)}"
                )
            elif "temporal_analysis" in analysis_result:
                temporal = analysis_result["temporal_analysis"]
                print(f"   Sequence Count: {temporal.get('sequence_count', 0)}")
                print(
                    f"   Suspicious Sequences: {temporal.get('suspicious_sequences', 0)}"
                )

            return True
        else:
            print(f"⚠️  {model_name} detailed analysis not supported or failed")
            print(f"   Status: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ {model_name} detailed analysis error: {e}")
        return False


def compare_model_results():
    """Compare results from different models."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)

    results = {}

    for model_name in MODELS_TO_TEST:
        print(f"\nTesting {model_name.upper()}...")

        try:
            start_time = time.time()

            with open(VIDEO_PATH, "rb") as f:
                files = {"video": (VIDEO_PATH, f, "video/mp4")}
                data = {
                    "video_id": f"comparison-{model_name}",
                    "model_name": model_name,
                }
                headers = {"X-API-Key": API_KEY}

                response = requests.post(
                    f"{API_URL}/analyze", files=files, data=data, headers=headers
                )

            request_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                analysis_result = result.get("result", {})

                results[model_name] = {
                    "prediction": analysis_result.get("prediction"),
                    "confidence": analysis_result.get("confidence", 0),
                    "processing_time": analysis_result.get("processing_time", 0),
                    "request_time": request_time,
                    "status": "SUCCESS",
                }
            else:
                results[model_name] = {"status": "FAILED", "error": response.text}

        except Exception as e:
            results[model_name] = {"status": "ERROR", "error": str(e)}

    # Display comparison
    print("\nRESULTS COMPARISON:")
    print("-" * 60)
    print(
        f"{'Model':<20} {'Prediction':<10} {'Confidence':<12} {'Proc.Time':<10} {'Status'}"
    )
    print("-" * 60)

    for model_name, result in results.items():
        if result["status"] == "SUCCESS":
            print(
                f"{model_name:<20} {result['prediction']:<10} {result['confidence']:<12.3f} "
                f"{result['processing_time']:<10.2f} {result['status']}"
            )
        else:
            print(
                f"{model_name:<20} {'N/A':<10} {'N/A':<12} {'N/A':<10} {result['status']}"
            )

    print("-" * 60)

    # Analysis
    successful_results = {k: v for k, v in results.items() if v["status"] == "SUCCESS"}

    if len(successful_results) > 1:
        predictions = [r["prediction"] for r in successful_results.values()]
        if len(set(predictions)) == 1:
            print(f"✅ All models agree: {predictions[0]}")
        else:
            print("⚠️  Models disagree on prediction!")
            for model, result in successful_results.items():
                print(
                    f"   {model}: {result['prediction']} ({result['confidence']:.3f})"
                )


def main():
    """Run comprehensive model tests."""
    print("🧪 Starting Comprehensive Model Testing...")
    print("Models to test:", ", ".join(MODELS_TO_TEST))
    print("=" * 70)

    if len(sys.argv) > 1:
        global VIDEO_PATH
        VIDEO_PATH = sys.argv[1]
        print(f"Using video: {VIDEO_PATH}")

    # Basic connectivity tests
    print("1. CONNECTIVITY TESTS")
    print("-" * 30)

    if not test_ping():
        print("💥 Server is not responding. Exiting.")
        sys.exit(1)

    if not test_health():
        print("💥 Server health check failed. Exiting.")
        sys.exit(1)

    print("\n2. INDIVIDUAL MODEL TESTS")
    print("-" * 30)

    # Test each model individually
    all_passed = True
    for model_name in MODELS_TO_TEST:
        print(f"\nTesting {model_name}:")
        if not test_analyze_with_model(model_name):
            all_passed = False

        # Test detailed analysis
        test_detailed_analysis_with_model(model_name)
        print()

    print("\n3. MODEL COMPARISON")
    print("-" * 30)

    # Compare models
    compare_model_results()

    print("\n" + "=" * 70)

    if all_passed:
        print("🎉 All model tests completed successfully!")
        print("\nNext steps:")
        print("1. Test with different video types")
        print("2. Performance benchmarking")
        print("3. Accuracy validation")
    else:
        print("💥 Some tests failed. Check the logs above.")
        print("\nTroubleshooting:")
        print("1. Verify all model files exist")
        print("2. Check server logs for errors")
        print("3. Ensure all dependencies are installed")


if __name__ == "__main__":
    main()
