#!/usr/bin/env python3
"""
Test script to verify device configuration is working properly.
Tests both CUDA and CPU device settings.
"""

import requests
import json

# Configuration
SERVER_URL = "http://localhost:8000"
API_KEY = "98e75012eea105151f4ddbab76872dd3717c74247903b302f74abdd783d0c5bb"

HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

def test_device_configuration():
    """Test device configuration endpoint."""
    print("🖥️  Testing Device Configuration")
    print("=" * 50)
    
    try:
        # Test device endpoint
        response = requests.get(f"{SERVER_URL}/device", headers=HEADERS, timeout=10)
        
        if response.status_code == 200:
            device_data = response.json()
            print(f"✅ Device Endpoint Success")
            print(f"📊 Device Type: {device_data.get('type', 'Unknown').upper()}")
            print(f"🏷️  Device Name: {device_data.get('name', 'Unknown')}")
            
            if device_data.get('type') == 'cuda':
                print(f"💾 Total VRAM: {device_data.get('total_memory', 0):.1f} GB")
                print(f"🔥 Used VRAM: {device_data.get('used_memory', 0):.1f} GB")
                print(f"📈 Memory Usage: {device_data.get('memory_usage_percent', 0):.1f}%")
                print(f"⚡ Compute Capability: {device_data.get('compute_capability', 'Unknown')}")
                print(f"🔧 CUDA Version: {device_data.get('cuda_version', 'Unknown')}")
            
            # Test configuration endpoint
            config_response = requests.get(f"{SERVER_URL}/config", headers=HEADERS, timeout=10)
            if config_response.status_code == 200:
                config_data = config_response.json()
                print(f"\n⚙️  Configuration:")
                print(f"🎯 Configured Device: {config_data.get('device', 'Unknown').upper()}")
                print(f"🚀 CUDA Available: {config_data.get('cuda_available', False)}")
                print(f"🔢 CUDA Devices: {config_data.get('cuda_device_count', 0)}")
                
                # Check consistency
                configured_device = config_data.get('device', '').lower()
                actual_device = device_data.get('type', '').lower()
                
                if configured_device == actual_device:
                    print(f"✅ Device configuration is consistent!")
                else:
                    print(f"⚠️  Device mismatch: Configured={configured_device}, Actual={actual_device}")
            
            return True
        else:
            print(f"❌ Device endpoint failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False

def main():
    """Run device configuration tests."""
    print("🧪 Testing Drishtiksha Device Configuration")
    print("=" * 60)
    
    success = test_device_configuration()
    
    if success:
        print("\n🎉 Device configuration test completed successfully!")
        print("\n💡 Tips:")
        print("   • Set DEVICE='cuda' in .env for GPU acceleration")
        print("   • Set DEVICE='cpu' in .env for CPU-only processing")
        print("   • Restart the server after changing device configuration")
    else:
        print("\n❌ Device configuration test failed!")

if __name__ == "__main__":
    main()
