#!/usr/bin/env python3
"""
Comprehensive Endpoint Testing Script
Tests all endpoints (/analyze, /analyze/detailed, /analyze/frames, /analyze/visualize)
for all models with load balancing validation.
"""

import os
import json
import time
import requests
from pathlib import Path

BASE_URL = "http://localhost:8000"
API_KEY = "98e75012eea105151f4ddbab76872dd3717c74247903b302f74abdd783d0c5bb"
HEADERS = {"X-API-Key": API_KEY}

# Test video file
TEST_VIDEO = "assets/id0_0001.mp4"


def test_endpoint(endpoint, model_name="", extra_data=None):
    """Test a specific endpoint with a model."""
    print(f"\n🧪 Testing {endpoint} with model: {model_name or 'default'}")

    if not os.path.exists(TEST_VIDEO):
        print(f"❌ Test video not found: {TEST_VIDEO}")
        return None

    files = {"video": open(TEST_VIDEO, "rb")}
    data = {"model_name": model_name}
    if extra_data:
        data.update(extra_data)

    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}{endpoint}", files=files, data=data, headers=HEADERS
        )
        duration = time.time() - start_time

        print(f"   Status: {response.status_code}")
        print(f"   Duration: {duration:.2f}s")

        if response.status_code == 200:
            if endpoint == "/analyze/visualize":
                print(f"   Response: Video file ({len(response.content)} bytes)")
                return {"success": True, "size": len(response.content)}
            else:
                result = response.json()
                print(f"   Success: {result.get('success', False)}")
                if "result" in result:
                    print(f"   Prediction: {result['result'].get('prediction', 'N/A')}")
                    print(f"   Confidence: {result['result'].get('confidence', 'N/A')}")
                return result
        else:
            try:
                error_detail = response.json()
                print(f"   Error: {error_detail}")
                return {"error": error_detail}
            except:
                print(f"   Error: {response.text}")
                return {"error": response.text}

    except Exception as e:
        print(f"   Exception: {e}")
        return {"exception": str(e)}
    finally:
        files["video"].close()


def test_load_balancing_status():
    """Test the load balancing status endpoint."""
    print("\n📊 Testing load balancing status...")
    try:
        response = requests.get(f"{BASE_URL}/status/load-balancing", headers=HEADERS)
        if response.status_code == 200:
            status = response.json()
            print(f"   Available models: {status.get('available_models', [])}")
            print(f"   Request counts: {status.get('request_counts', {})}")
            print(f"   Default model: {status.get('default_model', 'Unknown')}")
            return status
        else:
            print(f"   Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"   Exception: {e}")
        return None


def main():
    """Run comprehensive endpoint tests."""
    print("🚀 Starting Comprehensive Endpoint Testing")
    print("=" * 60)

    # Test server health
    print("\n🏥 Testing server health...")
    try:
        response = requests.get(f"{BASE_URL}/health", headers=HEADERS)
        if response.status_code == 200:
            health = response.json()
            print(f"   Status: {health.get('status', 'Unknown')}")
            print(f"   Model loaded: {health.get('model_loaded', False)}")
            print(f"   Model name: {health.get('model_name', 'Unknown')}")
        else:
            print(f"   Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"   Server not responding: {e}")
        return

    # Test load balancing status
    lb_status = test_load_balancing_status()
    available_models = (
        lb_status.get("available_models", [])
        if lb_status
        else ["siglip-lstm-v3", "color-cues-lstm-v1"]
    )

    print(f"\n🔧 Available models for testing: {available_models}")

    # All endpoints to test
    endpoints = [
        "/analyze",
        "/analyze/detailed",
        "/analyze/frames",
        "/analyze/visualize",
    ]

    results = {}

    # Test each endpoint with each model
    for endpoint in endpoints:
        print(f"\n{'=' * 20} TESTING {endpoint} {'=' * 20}")
        results[endpoint] = {}

        # Test with default model (load balanced)
        print(f"\n🎯 Testing {endpoint} with load balancing (no model specified)")
        result = test_endpoint(endpoint)
        results[endpoint]["default"] = result

        # Test with each specific model
        for model in available_models:
            print(f"\n🎯 Testing {endpoint} with {model}")
            result = test_endpoint(endpoint, model)
            results[endpoint][model] = result

    # Test load balancing after all requests
    print("\n" + "=" * 60)
    print("📊 FINAL LOAD BALANCING STATUS")
    print("=" * 60)
    final_status = test_load_balancing_status()

    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)

    for endpoint, model_results in results.items():
        print(f"\n{endpoint}:")
        for model, result in model_results.items():
            if result:
                if result.get("success") or result.get("size"):
                    print(f"   ✅ {model}: SUCCESS")
                elif "error" in result:
                    print(
                        f"   ❌ {model}: ERROR - {result.get('error', {}).get('error', 'Unknown error')}"
                    )
                else:
                    print(f"   ⚠️  {model}: UNKNOWN")
            else:
                print(f"   ❌ {model}: FAILED")

    # Check method support gracefully
    print(f"\n🔍 METHOD SUPPORT CHECK:")
    for endpoint, model_results in results.items():
        print(f"\n{endpoint}:")
        for model, result in model_results.items():
            if result and "error" in result:
                error_detail = result.get("error", {})
                if (
                    isinstance(error_detail, dict)
                    and "supported_methods" in error_detail
                ):
                    print(
                        f"   {model}: Missing method - Supported: {error_detail['supported_methods']}"
                    )
                elif isinstance(error_detail, dict) and "suggestion" in error_detail:
                    print(
                        f"   {model}: {error_detail.get('suggestion', 'No suggestion')}"
                    )
                else:
                    print(f"   {model}: Error occurred")
            else:
                print(f"   {model}: ✅ Method supported")


if __name__ == "__main__":
    main()
