#!/usr/bin/env python3
"""
Lightweight server health test script.
Tests server endpoints without requiring ML dependencies.
"""

import requests
import time
import sys


def test_ping():
    """Test the simple ping endpoint."""
    try:
        print("Testing ping endpoint...")
        response = requests.get("http://localhost:8000/ping", timeout=10)

        if response.status_code == 200:
            print("✅ Ping endpoint is working")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ Ping endpoint failed with status {response.status_code}")
            return False

    except requests.exceptions.Timeout:
        print("❌ Ping endpoint timed out")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server")
        return False
    except Exception as e:
        print(f"❌ Ping endpoint error: {e}")
        return False


def test_health():
    """Test the health check endpoint."""
    try:
        print("Testing health endpoint...")
        response = requests.get("http://localhost:8000/health", timeout=15)

        if response.status_code == 200:
            data = response.json()
            print("✅ Health endpoint is working")
            print(f"Status: {data.get('status')}")
            print(f"Model loaded: {data.get('model_loaded')}")
            print(f"Default model: {data.get('default_model')}")
            return True
        else:
            print(f"❌ Health endpoint failed with status {response.status_code}")
            return False

    except requests.exceptions.Timeout:
        print("❌ Health endpoint timed out")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server")
        return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False


def main():
    """Run all health tests."""
    print("🧪 Starting server health tests...")
    print("=" * 50)

    all_passed = True

    # Test basic connectivity
    if not test_ping():
        all_passed = False

    print("-" * 30)

    # Test health check
    if not test_health():
        all_passed = False

    print("=" * 50)

    if all_passed:
        print("🎉 All health tests passed! Server is responsive.")
        sys.exit(0)
    else:
        print("💥 Some health tests failed. Check server logs.")
        sys.exit(1)


if __name__ == "__main__":
    main()
