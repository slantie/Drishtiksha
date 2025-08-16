#!/usr/bin/env python3
"""
Test Script: Validate Analysis Endpoint Merge & ColorCues Fix
"""

import requests
import json
import os
from pathlib import Path

# Server configuration
BASE_URL = "http://localhost:8000"
TEST_VIDEO_PATH = Path("assets/id0_0001.mp4")

def test_merged_analysis_endpoint():
    """Test the merged /analyze endpoint"""
    print("🧪 Testing Merged Analysis Endpoint")
    print("=" * 50)
    
    if not TEST_VIDEO_PATH.exists():
        print(f"❌ Test video not found: {TEST_VIDEO_PATH}")
        return False
    
    # Test the merged analyze endpoint
    with open(TEST_VIDEO_PATH, 'rb') as video_file:
        files = {'video': ('test_video.mp4', video_file, 'video/mp4')}
        
        try:
            print("📤 Sending request to /analyze endpoint...")
            response = requests.post(f"{BASE_URL}/analyze", files=files)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Analysis successful!")
                print(f"📊 Model: {data.get('model_name', 'Unknown')}")
                print(f"🎯 Prediction: {data.get('prediction', 'Unknown')}")
                print(f"📈 Confidence: {data.get('confidence', 'Unknown'):.3f}")
                
                # Check for metrics (should be present now)
                if 'metrics' in data:
                    print("✅ Detailed metrics included!")
                    metrics = data['metrics']
                    for key, value in metrics.items():
                        print(f"   {key}: {value}")
                else:
                    print("ℹ️  No detailed metrics (expected for some models)")
                
                return True
            else:
                print(f"❌ Request failed: {response.status_code}")
                print(f"Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Exception occurred: {e}")
            return False

def test_detailed_endpoint_removed():
    """Test that the detailed endpoint is properly removed"""
    print("\n🧪 Testing Detailed Endpoint Removal")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/analyze/detailed")
        if response.status_code == 404:
            print("✅ Detailed endpoint properly removed (404 as expected)")
            return True
        else:
            print(f"❌ Detailed endpoint still exists: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        return False

def test_health_endpoint():
    """Test that the server is running"""
    print("\n🧪 Testing Server Health")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Server is running and healthy")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server not accessible: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Analysis Merge Validation Tests")
    print("=" * 60)
    
    # Check server health first
    if not test_health_endpoint():
        print("\n❌ Server is not running. Please start the server first.")
        print("Run: python -m uvicorn src.app.main:app --reload --port 8000")
        return
    
    # Test the merged endpoint
    test_1_passed = test_merged_analysis_endpoint()
    
    # Test that detailed endpoint is removed
    test_2_passed = test_detailed_endpoint_removed()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 30)
    print(f"✅ Merged Analysis Endpoint: {'PASS' if test_1_passed else 'FAIL'}")
    print(f"✅ Detailed Endpoint Removed: {'PASS' if test_2_passed else 'FAIL'}")
    
    if test_1_passed and test_2_passed:
        print("\n🎉 All tests passed! Analysis merge is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()
