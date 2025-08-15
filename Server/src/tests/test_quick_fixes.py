#!/usr/bin/env python3
"""
Quick test for fixed issues
"""

import requests
import time

BASE_URL = "http://localhost:8000"
API_KEY = "98e75012eea105151f4ddbab76872dd3717c74247903b302f74abdd783d0c5bb"
HEADERS = {"X-API-Key": API_KEY}
TEST_VIDEO = "assets/id0_0001.mp4"


def test_fixed_issues():
    print("🧪 Testing Fixed Issues")
    print("=" * 50)

    # Test SIGLIP-LSTM-V1 /analyze endpoint
    print("\n1. Testing SIGLIP-LSTM-V1 /analyze (should work now)")
    files = {"video": open(TEST_VIDEO, "rb")}
    data = {"model_name": "siglip-lstm-v1"}

    try:
        response = requests.post(
            f"{BASE_URL}/analyze", files=files, data=data, headers=HEADERS
        )
        if response.status_code == 200:
            result = response.json()
            print(
                f"   ✅ SUCCESS: {result['result']['prediction']} (conf: {result['result']['confidence']:.3f})"
            )
        else:
            print(f"   ❌ FAILED: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    finally:
        files["video"].close()

    # Test ColorCues /analyze/frames endpoint
    print("\n2. Testing ColorCues /analyze/frames (should work now)")
    files = {"video": open(TEST_VIDEO, "rb")}
    data = {"model_name": "color-cues-lstm-v1"}

    try:
        response = requests.post(
            f"{BASE_URL}/analyze/frames", files=files, data=data, headers=HEADERS
        )
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ SUCCESS: {result['summary']['prediction']} frames analysis")
        else:
            print(f"   ❌ FAILED: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    finally:
        files["video"].close()

    # Test load balancing status
    print("\n3. Testing Load Balancing Status")
    try:
        response = requests.get(f"{BASE_URL}/status/load-balancing", headers=HEADERS)
        if response.status_code == 200:
            status = response.json()
            print(f"   ✅ Request counts: {status['request_counts']}")
        else:
            print(f"   ❌ FAILED: {response.status_code}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")


if __name__ == "__main__":
    test_fixed_issues()
