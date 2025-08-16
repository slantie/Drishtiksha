# test_e2e_api.py

import os
import time
import requests
from typing import List

# ===================================================================
# --- CONFIGURABLE SETTINGS ---
# ===================================================================
BASE_URL = "http://localhost:8000"
API_KEY = "98e75012eea105151f4ddbab76872dd3717c74247903b302f74abdd783d0c5bb"
VIDEO_PATH = "assets/id0_0001.mp4"

# List of models to test against the endpoints
MODELS_TO_TEST: List[str] = [
    "SIGLIP-LSTM-V1",
    "SIGLIP-LSTM-V3",
    "COLOR-CUES-LSTM-V1",
]

MODELS_WITH_ADVANCED_FEATURES: List[str] = [
    "SIGLIP-LSTM-V3",
    "COLOR-CUES-LSTM-V1",
]

def print_header(title: str):
    """Prints a formatted header."""
    print("\n" + "="*60)
    print(f"📋 {title.upper()}")
    print("="*60)

def print_result(test_name: str, success: bool, duration: float, details: str = ""):
    status = "✅ PASSED" if success else "❌ FAILED"
    print(f"{status:<15} | {test_name:<45} | Time: {duration:.2f}s")
    if details:
        print(f"    └── Details: {details}")

def test_status_endpoints():
    print_header("Testing Status Endpoints")
    start_time = time.time()
    try:
        response = requests.get(f"{BASE_URL}/")
        response.raise_for_status()
        data = response.json()
        assert data["status"] == "ok"
        print_result("GET / (Health Check)", True, time.time() - start_time, f"Default model: {data['default_model']}")
    except Exception as e:
        print_result("GET / (Health Check)", False, time.time() - start_time, str(e))

def test_analysis_endpoints():
    print_header("Testing Analysis Endpoints")
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ CRITICAL: Test video not found at '{VIDEO_PATH}'. Aborting tests.")
        return
    headers = {"X-API-Key": API_KEY}
    for model in MODELS_TO_TEST:
        print(f"\n--- Testing with model: [{model}] ---")
        endpoints = [
            ("/analyze", "POST /analyze (quick)"),
            ("/analyze/detailed", "POST /detailed"),
            ("/analyze/frames", "POST /frames"),
            ("/analyze/visualize", "POST /visualize")
        ]
        for endpoint, test_name in endpoints:
            start_time = time.time()
            try:
                with open(VIDEO_PATH, 'rb') as f:
                    files = {'video': (os.path.basename(VIDEO_PATH), f, 'video/mp4')}
                    payload = {'model': model}
                    response = requests.post(f"{BASE_URL}{endpoint}", headers=headers, files=files, data=payload)
                    is_advanced = model in MODELS_WITH_ADVANCED_FEATURES and endpoint != "/analyze"
                    is_simple = endpoint == "/analyze"
                    
                    if is_simple or is_advanced:
                        response.raise_for_status()
                        if "visualize" in endpoint:
                            details = f"Received video of size {len(response.content) / 1024:.2f} KB"
                        else:
                            data = response.json()
                            details = f"Prediction: {data['data'].get('prediction', data['data'].get('overall_prediction'))}"
                        print_result(test_name, True, time.time() - start_time, details)
                    else:
                        assert response.status_code == 501
                        print_result(test_name, True, time.time() - start_time, "Received expected 501 error.")
            except requests.exceptions.HTTPError as e:
                details = str(e)
                try:
                    error_details = e.response.json()
                    details += f" | Server Response: {error_details}"
                except: pass
                print_result(test_name, False, time.time() - start_time, details)
            except Exception as e:
                print_result(test_name, False, time.time() - start_time, str(e))

if __name__ == "__main__":
    test_status_endpoints()
    test_analysis_endpoints()
    print_header("All Tests Completed")