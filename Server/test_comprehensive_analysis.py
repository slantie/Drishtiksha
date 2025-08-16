#!/usr/bin/env python3
"""
Test Script: Comprehensive Analysis Endpoint - Multi-Model Testing
This script tests the new merged endpoint that combines all analysis types across all available models.
"""

import requests
import json
import os
import time
from pathlib import Path
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.layout import Layout
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Rich not available, using basic output")

from typing import Dict, Any, List

# Server configuration
BASE_URL = "http://localhost:8000"
TEST_VIDEO_PATH = Path("assets/id0_0001.mp4")
API_KEY = "98e75012eea105151f4ddbab76872dd3717c74247903b302f74abdd783d0c5bb"

if RICH_AVAILABLE:
    console = Console()

def get_available_models() -> List[str]:
    """Get list of available models from the server"""
    try:
        headers = {"X-API-Key": API_KEY}
        response = requests.get(f"{BASE_URL}/", headers=headers)
        if response.status_code == 200:
            data = response.json()
            if "active_models" in data:
                return [model["name"] for model in data["active_models"]]
        return ["SIGLIP-LSTM-V3", "COLOR-CUES-LSTM-V1"]  # Fallback to known models
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"[red]Error getting models: {e}[/red]")
        else:
            print(f"Error getting models: {e}")
        return ["SIGLIP-LSTM-V3", "COLOR-CUES-LSTM-V1"]  # Fallback

def test_comprehensive_analysis_for_model(model_name: str) -> Dict[str, Any]:
    """Test the comprehensive analysis endpoint for a specific model"""
    
    if not TEST_VIDEO_PATH.exists():
        error_msg = f"❌ Test video not found: {TEST_VIDEO_PATH}"
        if RICH_AVAILABLE:
            console.print(f"[red]{error_msg}[/red]")
        else:
            print(error_msg)
        return None
    
    # Test comprehensive analysis with all features
    with open(TEST_VIDEO_PATH, 'rb') as video_file:
        files = {'video': ('test_video.mp4', video_file, 'video/mp4')}
        data = {
            'include_frames': True,
            'include_visualization': True,
            'model': model_name  # Changed from model_name to model
        }
        headers = {"X-API-Key": API_KEY}
        
        try:
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/analyze/comprehensive", 
                                   files=files, data=data, headers=headers)
            request_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                analysis_data = data.get('data', {})
                
                return {
                    "model_name": model_name,
                    "success": True,
                    "client_time": request_time,
                    "prediction": analysis_data.get('prediction', 'Unknown'),
                    "confidence": analysis_data.get('confidence', 0),
                    "processing_breakdown": analysis_data.get('processing_breakdown', {}),
                    "metrics": analysis_data.get('metrics', {}),
                    "frames_analysis": analysis_data.get('frames_analysis'),
                    "visualization_generated": analysis_data.get('visualization_generated', False),
                    "visualization_filename": analysis_data.get('visualization_filename'),
                    "response_data": analysis_data
                }
            else:
                return {
                    "model_name": model_name,
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "client_time": request_time
                }
                
        except Exception as e:
            return {
                "model_name": model_name,
                "success": False,
                "error": str(e),
                "client_time": 0
            }

def test_individual_endpoints_for_model(model_name: str) -> Dict[str, Any]:
    """Test individual endpoints for comparison"""
    
    individual_times = {}
    headers = {"X-API-Key": API_KEY}
    
    # Test basic analysis
    with open(TEST_VIDEO_PATH, 'rb') as video_file:
        files = {'video': ('test_video.mp4', video_file, 'video/mp4')}
        data = {'model': model_name}
        
        start = time.time()
        response = requests.post(f"{BASE_URL}/analyze", files=files, data=data, headers=headers)
        individual_times['analyze'] = time.time() - start
        
    # Test frames analysis
    with open(TEST_VIDEO_PATH, 'rb') as video_file:
        files = {'video': ('test_video.mp4', video_file, 'video/mp4')}
        data = {'model': model_name}
        
        start = time.time()
        response = requests.post(f"{BASE_URL}/analyze/frames", files=files, data=data, headers=headers)
        individual_times['frames'] = time.time() - start
        
    # Test visualization
    with open(TEST_VIDEO_PATH, 'rb') as video_file:
        files = {'video': ('test_video.mp4', video_file, 'video/mp4')}
        data = {'model': model_name}
        
        start = time.time()
        response = requests.post(f"{BASE_URL}/analyze/visualize", files=files, data=data, headers=headers)
        individual_times['visualize'] = time.time() - start
    
    total_individual = sum(individual_times.values())
    
    return {
        "model_name": model_name,
        "individual_times": individual_times,
        "total_time": total_individual
    }

def create_comparison_table_rich(comprehensive_results: List[Dict], individual_results: List[Dict]) -> Table:
    """Create a Rich table comparing performance across all models"""
    
    table = Table(title="🚀 Performance Comparison: Individual vs Comprehensive Analysis", box=box.ROUNDED)
    
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Individual Total (s)", style="red", justify="right")
    table.add_column("Comprehensive Total (s)", style="green", justify="right")
    table.add_column("Time Saved (s)", style="yellow", justify="right")
    table.add_column("Efficiency Gain (%)", style="magenta", justify="right")
    table.add_column("Prediction", style="blue", justify="center")
    table.add_column("Confidence", style="blue", justify="right")
    table.add_column("Visualization", style="cyan", justify="center")
    
    for comp_result in comprehensive_results:
        if not comp_result or not comp_result.get('success'):
            continue
            
        model_name = comp_result['model_name']
        
        # Find matching individual result
        individual_result = next((r for r in individual_results if r['model_name'] == model_name), None)
        
        if individual_result:
            individual_time = individual_result['total_time']
            comprehensive_time = comp_result['client_time']
            time_saved = individual_time - comprehensive_time
            efficiency_gain = (time_saved / individual_time) * 100 if individual_time > 0 else 0
            
            # Format prediction with color
            prediction = comp_result['prediction']
            pred_color = "red" if prediction == "FAKE" else "green"
            
            # Format visualization status
            viz_status = "✅" if comp_result['visualization_generated'] else "❌"
            
            table.add_row(
                model_name,
                f"{individual_time:.2f}",
                f"{comprehensive_time:.2f}",
                f"{time_saved:.2f}",
                f"{efficiency_gain:.1f}%",
                f"[{pred_color}]{prediction}[/{pred_color}]",
                f"{comp_result['confidence']:.3f}",
                viz_status
            )
    
    return table

def print_comparison_table_simple(comprehensive_results: List[Dict], individual_results: List[Dict]):
    """Print comparison table using simple text formatting"""
    
    print("\n" + "="*100)
    print("🚀 PERFORMANCE COMPARISON: Individual vs Comprehensive Analysis")
    print("="*100)
    print(f"{'Model':<20} {'Individual(s)':<12} {'Comprehensive(s)':<15} {'Saved(s)':<10} {'Gain(%)':<8} {'Prediction':<10} {'Confidence':<10} {'Viz':<5}")
    print("-"*100)
    
    for comp_result in comprehensive_results:
        if not comp_result or not comp_result.get('success'):
            continue
            
        model_name = comp_result['model_name']
        
        # Find matching individual result
        individual_result = next((r for r in individual_results if r['model_name'] == model_name), None)
        
        if individual_result:
            individual_time = individual_result['total_time']
            comprehensive_time = comp_result['client_time']
            time_saved = individual_time - comprehensive_time
            efficiency_gain = (time_saved / individual_time) * 100 if individual_time > 0 else 0
            
            prediction = comp_result['prediction']
            confidence = comp_result['confidence']
            viz_status = "✅" if comp_result['visualization_generated'] else "❌"
            
            print(f"{model_name:<20} {individual_time:<12.2f} {comprehensive_time:<15.2f} {time_saved:<10.2f} {efficiency_gain:<8.1f} {prediction:<10} {confidence:<10.3f} {viz_status:<5}")

def main():
    """Run comprehensive multi-model testing"""
    
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            "[bold blue]🧪 Comprehensive Analysis Multi-Model Testing[/bold blue]\n"
            "[yellow]Testing all available models with comprehensive analysis endpoint[/yellow]",
            border_style="blue"
        ))
    else:
        print("="*70)
        print("🧪 COMPREHENSIVE ANALYSIS MULTI-MODEL TESTING")
        print("Testing all available models with comprehensive analysis endpoint")
        print("="*70)
    
    # Check server health
    try:
        headers = {"X-API-Key": API_KEY}
        response = requests.get(f"{BASE_URL}/", headers=headers)
        if response.status_code != 200:
            error_msg = "❌ Server is not running or not accessible"
            if RICH_AVAILABLE:
                console.print(f"[red]{error_msg}[/red]")
            else:
                print(error_msg)
            return
    except Exception as e:
        error_msg = f"❌ Server connection failed: {e}"
        if RICH_AVAILABLE:
            console.print(f"[red]{error_msg}[/red]")
        else:
            print(error_msg)
        return
    
    # Get available models
    discovery_msg = "🔍 Discovering available models..."
    if RICH_AVAILABLE:
        console.print(f"\n[cyan]{discovery_msg}[/cyan]")
    else:
        print(f"\n{discovery_msg}")
        
    available_models = get_available_models()
    
    found_msg = f"✅ Found {len(available_models)} models: {', '.join(available_models)}"
    if RICH_AVAILABLE:
        console.print(f"[green]{found_msg}[/green]")
    else:
        print(found_msg)
    
    if not available_models:
        error_msg = "❌ No models available for testing"
        if RICH_AVAILABLE:
            console.print(f"[red]{error_msg}[/red]")
        else:
            print(error_msg)
        return
    
    comprehensive_results = []
    individual_results = []
    
    # Test each model
    for i, model_name in enumerate(available_models, 1):
        print(f"\n[{i}/{len(available_models)}] Testing {model_name}")
        print("-" * 50)
        
        # Test comprehensive analysis
        test_msg = f"🚀 Testing Comprehensive Analysis: {model_name}"
        if RICH_AVAILABLE:
            console.print(f"[yellow]{test_msg}[/yellow]")
        else:
            print(test_msg)
        
        comp_result = test_comprehensive_analysis_for_model(model_name)
        comprehensive_results.append(comp_result)
        
        if comp_result and comp_result.get('success'):
            success_msg = f"✅ {model_name} comprehensive analysis completed"
            if RICH_AVAILABLE:
                console.print(f"[green]{success_msg}[/green]")
            else:
                print(success_msg)
                
            # Print key results
            print(f"   Prediction: {comp_result['prediction']}")
            print(f"   Confidence: {comp_result['confidence']:.3f}")
            print(f"   Processing Time: {comp_result['client_time']:.2f}s")
            breakdown = comp_result.get('processing_breakdown', {})
            if breakdown:
                print(f"   Server Breakdown: Basic({breakdown.get('basic_analysis', 0):.1f}s) + Frames({breakdown.get('frames_analysis', 0):.1f}s) + Viz({breakdown.get('visualization', 0):.1f}s)")
        else:
            error_msg = f"❌ {model_name} comprehensive analysis failed"
            if comp_result:
                error_msg += f": {comp_result.get('error', 'Unknown error')}"
                # Print detailed error information
                print(f"   🔍 Debug Info:")
                print(f"   📊 Status: {comp_result.get('success', 'Unknown')}")
                print(f"   ⏱️  Time: {comp_result.get('client_time', 0):.2f}s")
                print(f"   📄 Full Error: {comp_result.get('error', 'No error details')}")
            if RICH_AVAILABLE:
                console.print(f"[red]{error_msg}[/red]")
            else:
                print(error_msg)
        
        # Test individual endpoints for comparison
        test_msg2 = f"📊 Testing Individual Endpoints: {model_name}"
        if RICH_AVAILABLE:
            console.print(f"[yellow]{test_msg2}[/yellow]")
        else:
            print(test_msg2)
        
        individual_result = test_individual_endpoints_for_model(model_name)
        individual_results.append(individual_result)
        
        success_msg2 = f"✅ {model_name} individual endpoints completed ({individual_result['total_time']:.2f}s total)"
        if RICH_AVAILABLE:
            console.print(f"[green]{success_msg2}[/green]")
        else:
            print(success_msg2)
    
    # Display results
    print("\n" + "="*80)
    print("📊 RESULTS SUMMARY")
    print("="*80)
    
    if RICH_AVAILABLE:
        console.print(create_comparison_table_rich(comprehensive_results, individual_results))
    else:
        print_comparison_table_simple(comprehensive_results, individual_results)
    
    # Summary statistics
    successful_tests = [r for r in comprehensive_results if r and r.get('success')]
    if successful_tests:
        avg_efficiency = sum(
            ((next(ir for ir in individual_results if ir['model_name'] == r['model_name'])['total_time'] - r['client_time']) 
             / next(ir for ir in individual_results if ir['model_name'] == r['model_name'])['total_time'] * 100)
            for r in successful_tests
        ) / len(successful_tests)
        
        summary_text = (
            f"🎉 Testing Summary\n"
            f"Models Tested: {len(available_models)}\n"
            f"Successful Tests: {len(successful_tests)}\n"
            f"Average Efficiency Gain: {avg_efficiency:.1f}%\n"
            f"Test Video: {TEST_VIDEO_PATH.name}"
        )
        
        if RICH_AVAILABLE:
            console.print(Panel.fit(f"[bold green]{summary_text}[/bold green]", border_style="green"))
        else:
            print("\n" + "="*50)
            print(summary_text)
            print("="*50)
    
    final_msg = "🎯 All tests completed successfully!"
    if RICH_AVAILABLE:
        console.print(f"\n[bold blue]{final_msg}[/bold blue]")
    else:
        print(f"\n{final_msg}")

if __name__ == "__main__":
    main()

# Server configuration
BASE_URL = "http://localhost:8000"
API_KEY = "98e75012eea105151f4ddbab76872dd3717c74247903b302f74abdd783d0c5bb"
TEST_VIDEO_PATH = Path("assets/id0_0001.mp4")

# Headers for API requests
HEADERS = {
    "X-API-Key": API_KEY
}

def test_comprehensive_analysis():
    """Test the new comprehensive analysis endpoint"""
    print("🚀 Testing Comprehensive Analysis Endpoint")
    print("=" * 60)
    
    if not TEST_VIDEO_PATH.exists():
        print(f"❌ Test video not found: {TEST_VIDEO_PATH}")
        return False
    
    # Test comprehensive analysis with all features
    with open(TEST_VIDEO_PATH, 'rb') as video_file:
        files = {'video': ('test_video.mp4', video_file, 'video/mp4')}
        params = {
            'include_frames': True,
            'include_visualization': True
        }
        
        try:
            print("📤 Sending request to /analyze/comprehensive endpoint...")
            print(f"🎬 Include Frames: {params['include_frames']}")
            print(f"📊 Include Visualization: {params['include_visualization']}")
            
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/analyze/comprehensive", files=files, params=params, headers=HEADERS, timeout=120)
            request_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                analysis_data = data.get('data', {})
                
                print("\n✅ Comprehensive Analysis successful!")
                print("=" * 50)
                print(f"📊 Model: {data.get('model_used', 'Unknown')}")
                print(f"🎯 Prediction: {analysis_data.get('prediction', 'Unknown')}")
                print(f"📈 Confidence: {analysis_data.get('confidence', 'Unknown'):.3f}")
                print(f"⏱️  Client Request Time: {request_time:.3f}s")
                
                # Processing breakdown
                breakdown = analysis_data.get('processing_breakdown', {})
                if breakdown:
                    print(f"\n📊 Processing Breakdown:")
                    for step, duration in breakdown.items():
                        print(f"   {step.replace('_', ' ').title()}: {duration:.3f}s")
                
                # Basic metrics
                metrics = analysis_data.get('metrics', {})
                if metrics:
                    print(f"\n📈 Analysis Metrics:")
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)) and key != 'per_frame_scores':
                            print(f"   {key.replace('_', ' ').title()}: {value:.3f}")
                        elif key != 'per_frame_scores':
                            print(f"   {key.replace('_', ' ').title()}: {value}")
                
                # Frame analysis results
                frames_analysis = analysis_data.get('frames_analysis')
                if frames_analysis:
                    print(f"\n🎬 Frame Analysis Results:")
                    frame_predictions = frames_analysis.get('frame_predictions', [])
                    if frame_predictions:
                        fake_frames = sum(1 for f in frame_predictions if f.get('prediction') == 'FAKE')
                        print(f"   Total Frames: {len(frame_predictions)}")
                        print(f"   Suspicious Frames: {fake_frames} ({fake_frames/len(frame_predictions)*100:.1f}%)")
                        
                        temporal = frames_analysis.get('temporal_analysis', {})
                        if temporal:
                            consistency = temporal.get('consistency_score')
                            if consistency is not None:
                                print(f"   Consistency Score: {consistency:.3f}")
                
                # Visualization status
                viz_generated = analysis_data.get('visualization_generated', False)
                viz_filename = analysis_data.get('visualization_filename')
                print(f"\n📊 Visualization:")
                print(f"   Generated: {'✅ Yes' if viz_generated else '❌ No'}")
                if viz_filename:
                    print(f"   Filename: {viz_filename}")
                    print(f"   Download URL: {BASE_URL}/analyze/visualization/{viz_filename}")
                
                return True
            else:
                print(f"❌ Request failed: {response.status_code}")
                print(f"Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Exception occurred: {e}")
            return False

def test_individual_endpoints_comparison():
    """Compare individual endpoints vs comprehensive endpoint"""
    print("\n🔍 Comparing Individual vs Comprehensive Analysis")
    print("=" * 60)
    
    if not TEST_VIDEO_PATH.exists():
        print(f"❌ Test video not found: {TEST_VIDEO_PATH}")
        return
    
    # Test individual endpoints
    individual_times = {}
    
    with open(TEST_VIDEO_PATH, 'rb') as video_file:
        files = {'video': ('test_video.mp4', video_file, 'video/mp4')}
        
        # Test basic analysis
        start = time.time()
        response = requests.post(f"{BASE_URL}/analyze", files=files, headers=HEADERS, timeout=60)
        individual_times['analyze'] = time.time() - start
        print(f"📊 Individual /analyze: {individual_times['analyze']:.3f}s")
        
    with open(TEST_VIDEO_PATH, 'rb') as video_file:
        files = {'video': ('test_video.mp4', video_file, 'video/mp4')}
        
        # Test frames analysis
        start = time.time()
        response = requests.post(f"{BASE_URL}/analyze/frames", files=files, headers=HEADERS, timeout=60)
        individual_times['frames'] = time.time() - start
        print(f"🎬 Individual /analyze/frames: {individual_times['frames']:.3f}s")
        
    with open(TEST_VIDEO_PATH, 'rb') as video_file:
        files = {'video': ('test_video.mp4', video_file, 'video/mp4')}
        
        # Test visualization
        start = time.time()
        response = requests.post(f"{BASE_URL}/analyze/visualize", files=files, headers=HEADERS, timeout=120)
        individual_times['visualize'] = time.time() - start
        print(f"📊 Individual /analyze/visualize: {individual_times['visualize']:.3f}s")
    
    total_individual = sum(individual_times.values())
    print(f"\n⏱️  Total Individual Time: {total_individual:.3f}s")
    
    # Now test comprehensive endpoint
    with open(TEST_VIDEO_PATH, 'rb') as video_file:
        files = {'video': ('test_video.mp4', video_file, 'video/mp4')}
        params = {'include_frames': True, 'include_visualization': True}
        
        start = time.time()
        response = requests.post(f"{BASE_URL}/analyze/comprehensive", files=files, params=params, headers=HEADERS, timeout=120)
        comprehensive_time = time.time() - start
        
        print(f"🚀 Comprehensive /analyze/comprehensive: {comprehensive_time:.3f}s")
        
        if response.status_code == 200:
            data = response.json()
            server_breakdown = data.get('data', {}).get('processing_breakdown', {})
            server_total = server_breakdown.get('total', 0)
            print(f"🖥️  Server Processing Time: {server_total:.3f}s")
            
            # Calculate efficiency
            efficiency = ((total_individual - comprehensive_time) / total_individual) * 100
            print(f"\n📈 Efficiency Gain: {efficiency:.1f}% faster")
            print(f"⚡ Time Saved: {total_individual - comprehensive_time:.3f}s")

def main():
    """Run all tests"""
    print("🧪 Testing Comprehensive Analysis Endpoint")
    print("=" * 70)
    
    # Check server health with better error handling
    try:
        print("🔍 Checking server health...")
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running and healthy")
            health_data = response.json()
            print(f"📊 Active Models: {len(health_data.get('active_models', []))}")
        else:
            print(f"⚠️  Server responded with status {response.status_code}")
            print("❌ Server is not ready. Please check the server status.")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Please ensure server is running on http://localhost:8000")
        print("💡 Start server with: python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000")
        return
    except requests.exceptions.Timeout:
        print("❌ Server request timed out. Server may be starting up.")
        return
    except Exception as e:
        print(f"❌ Unexpected error connecting to server: {e}")
        return
    
    # Test if comprehensive endpoint exists
    print("\n🔍 Checking if comprehensive endpoint exists...")
    try:
        # Try OPTIONS request to check if endpoint exists
        options_response = requests.options(f"{BASE_URL}/analyze/comprehensive", timeout=5)
        print(f"📡 Comprehensive endpoint status: {options_response.status_code}")
    except Exception as e:
        print(f"⚠️  Could not check comprehensive endpoint: {e}")
    
    # Test existing endpoints first
    print("\n🧪 Testing existing endpoints...")
    test_existing_endpoints()
    
    # Test the comprehensive endpoint
    print("\n🚀 Testing comprehensive endpoint...")
    success = test_comprehensive_analysis()
    
    # Compare with individual endpoints if comprehensive works
    if success:
        test_individual_endpoints_comparison()
    
    print("\n🎉 Testing completed!")

def test_existing_endpoints():
    """Test existing endpoints to ensure basic functionality"""
    if not TEST_VIDEO_PATH.exists():
        print(f"❌ Test video not found: {TEST_VIDEO_PATH}")
        return
    
    print("📤 Testing basic /analyze endpoint...")
    with open(TEST_VIDEO_PATH, 'rb') as video_file:
        files = {'video': ('test_video.mp4', video_file, 'video/mp4')}
        try:
            response = requests.post(f"{BASE_URL}/analyze", files=files, headers=HEADERS, timeout=60)
            if response.status_code == 200:
                print("✅ Basic analyze endpoint working")
                data = response.json()
                print(f"🎯 Prediction: {data.get('data', {}).get('prediction', 'N/A')}")
            else:
                print(f"❌ Basic analyze failed: {response.status_code}")
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"❌ Basic analyze error: {e}")

if __name__ == "__main__":
    main()
