#!/usr/bin/env python3
"""
Test script for the new server statistics endpoints.
Verifies that all monitoring endpoints are working correctly.
"""

import requests
import json
import time
from typing import Dict, Any

# Configuration
SERVER_URL = "http://localhost:8000"
API_KEY = "98e75012eea105151f4ddbab76872dd3717c74247903b302f74abdd783d0c5bb"

HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

def test_endpoint(endpoint: str, expected_fields: list = None) -> Dict[str, Any]:
    """Test a specific endpoint and return the response."""
    url = f"{SERVER_URL}{endpoint}"
    print(f"\n🔍 Testing: {endpoint}")
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success: {response.status_code}")
            
            # Check for expected fields if provided
            if expected_fields:
                missing_fields = [field for field in expected_fields if field not in data]
                if missing_fields:
                    print(f"⚠️  Missing fields: {missing_fields}")
                else:
                    print(f"✅ All expected fields present: {expected_fields}")
            
            return data
        else:
            print(f"❌ Failed: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return None

def main():
    """Run tests for all monitoring endpoints."""
    print("🚀 Testing Drishtiksha Server Statistics Endpoints")
    print("=" * 60)
    
    # Test basic health check
    health_data = test_endpoint("/", ["status", "active_models", "default_model"])
    
    # Test ping
    ping_data = test_endpoint("/ping", ["status"])
    
    # Test comprehensive stats
    stats_data = test_endpoint("/stats", [
        "service_name", "version", "status", "uptime_seconds",
        "device_info", "system_info", "models_info", 
        "active_models_count", "total_models_count", "configuration"
    ])
    
    # Test device info
    device_data = test_endpoint("/device", ["type", "name"])
    
    # Test system info
    system_data = test_endpoint("/system", [
        "python_version", "platform", "cpu_count", 
        "total_ram", "used_ram", "ram_usage_percent", "uptime_seconds"
    ])
    
    # Test models info
    models_data = test_endpoint("/models", ["models", "summary"])
    
    # Test configuration
    config_data = test_endpoint("/config", [
        "project_name", "device", "default_model", "active_models"
    ])
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    if stats_data:
        print(f"🏃 Service: {stats_data.get('service_name', 'Unknown')} v{stats_data.get('version', 'Unknown')}")
        print(f"⏱️  Uptime: {stats_data.get('uptime_seconds', 0):.1f} seconds")
        
        device_info = stats_data.get('device_info', {})
        print(f"🖥️  Device: {device_info.get('type', 'unknown').upper()}")
        
        if device_info.get('type') == 'cuda':
            print(f"🔥 GPU: {device_info.get('name', 'Unknown')}")
            print(f"💾 VRAM: {device_info.get('used_memory', 0):.1f}GB / {device_info.get('total_memory', 0):.1f}GB ({device_info.get('memory_usage_percent', 0):.1f}%)")
        
        system_info = stats_data.get('system_info', {})
        print(f"🐍 Python: {system_info.get('python_version', 'Unknown')}")
        print(f"💻 RAM: {system_info.get('used_ram', 0):.1f}GB / {system_info.get('total_ram', 0):.1f}GB ({system_info.get('ram_usage_percent', 0):.1f}%)")
        
        print(f"🤖 Models: {stats_data.get('active_models_count', 0)} active / {stats_data.get('total_models_count', 0)} total")
        
        config = stats_data.get('configuration', {})
        active_models = config.get('active_models', [])
        print(f"✅ Active Models: {', '.join(active_models)}")
    
    print("\n🎉 Testing completed!")

if __name__ == "__main__":
    main()
