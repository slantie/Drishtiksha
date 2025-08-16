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
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.layout import Layout
from rich.text import Text
from rich import box
from typing import Dict, Any, List

# Server configuration
BASE_URL = "http://localhost:8000"
TEST_VIDEO_PATH = Path("assets/id0_0001.mp4")
API_KEY = "98e75012eea105151f4ddbab76872dd3717c74247903b302f74abdd783d0c5bb"

# Initialize Rich console
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
        console.print(f"[red]Error getting models: {e}[/red]")
        return ["SIGLIP-LSTM-V3", "COLOR-CUES-LSTM-V1"]  # Fallback

def test_comprehensive_analysis_for_model(model_name: str) -> Dict[str, Any]:
    """Test the comprehensive analysis endpoint for a specific model"""
    
    if not TEST_VIDEO_PATH.exists():
        console.print(f"[red]❌ Test video not found: {TEST_VIDEO_PATH}[/red]")
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
                    "error": f"HTTP {response.status_code}: {response.text[:500]}",  # Limit error text
                    "client_time": request_time,
                    "status_code": response.status_code,
                    "response_text": response.text[:1000]  # Add full response for debugging
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

def create_performance_comparison_table(comprehensive_results: List[Dict], individual_results: List[Dict]) -> Table:
    """Create a Rich table comparing performance across all models"""
    
    table = Table(title="🚀 Performance Comparison: Individual vs Comprehensive Analysis", box=box.ROUNDED)
    
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Individual\nTotal (s)", style="red", justify="right")
    table.add_column("Comprehensive\nTotal (s)", style="green", justify="right")
    table.add_column("Time Saved\n(s)", style="yellow", justify="right")
    table.add_column("Efficiency\nGain (%)", style="magenta", justify="right")
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

def create_processing_breakdown_table(comprehensive_results: List[Dict]) -> Table:
    """Create a detailed processing breakdown table"""
    
    table = Table(title="⚡ Processing Breakdown by Model", box=box.ROUNDED)
    
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Basic Analysis\n(s)", style="green", justify="right")
    table.add_column("Frames Analysis\n(s)", style="blue", justify="right")
    table.add_column("Visualization\n(s)", style="red", justify="right")
    table.add_column("Total Server\n(s)", style="yellow", justify="right")
    table.add_column("Analysis Units", style="magenta", justify="right")
    table.add_column("Suspicious\nUnits", style="red", justify="right")
    
    for result in comprehensive_results:
        if not result or not result.get('success'):
            continue
            
        breakdown = result.get('processing_breakdown', {})
        metrics = result.get('metrics', {})
        
        # Get processing times
        basic_time = breakdown.get('basic_analysis', 0)
        frames_time = breakdown.get('frames_analysis', 0)
        viz_time = breakdown.get('visualization', 0)
        total_time = breakdown.get('total', 0)
        
        # Get analysis metrics
        analysis_units = metrics.get('frame_count') or metrics.get('sequence_count', 'N/A')
        suspicious_units = metrics.get('suspicious_frames_count') or metrics.get('suspicious_sequences_count', 'N/A')
        
        table.add_row(
            result['model_name'],
            f"{basic_time:.2f}",
            f"{frames_time:.2f}",
            f"{viz_time:.2f}",
            f"{total_time:.2f}",
            str(analysis_units),
            str(suspicious_units)
        )
    
    return table

def create_model_metrics_table(comprehensive_results: List[Dict]) -> Table:
    """Create a table showing detailed model metrics"""
    
    table = Table(title="📊 Model Analysis Metrics", box=box.ROUNDED)
    
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Prediction", style="blue", justify="center")
    table.add_column("Confidence", style="green", justify="right")
    table.add_column("Avg Score", style="yellow", justify="right")
    table.add_column("Max Score", style="red", justify="right")
    table.add_column("Min Score", style="blue", justify="right")
    table.add_column("Total Frames", style="magenta", justify="right")
    table.add_column("Suspicious %", style="red", justify="right")
    
    for result in comprehensive_results:
        if not result or not result.get('success'):
            continue
            
        metrics = result.get('metrics', {})
        frames_analysis = result.get('frames_analysis', {})
        
        # Get basic metrics
        prediction = result['prediction']
        confidence = result['confidence']
        
        # Get detailed metrics
        avg_score = metrics.get('final_average_score', 'N/A')
        max_score = metrics.get('max_score', 'N/A')
        min_score = metrics.get('min_score', 'N/A')
        
        # Get frame analysis data
        frame_predictions = frames_analysis.get('frame_predictions', []) if frames_analysis else []
        total_frames = len(frame_predictions)
        suspicious_frames = sum(1 for f in frame_predictions if f.get('prediction') == 'FAKE')
        suspicious_percentage = (suspicious_frames / total_frames * 100) if total_frames > 0 else 0
        
        # Format prediction with color
        pred_color = "red" if prediction == "FAKE" else "green"
        
        table.add_row(
            result['model_name'],
            f"[{pred_color}]{prediction}[/{pred_color}]",
            f"{confidence:.3f}",
            f"{avg_score:.3f}" if isinstance(avg_score, (int, float)) else str(avg_score),
            f"{max_score:.3f}" if isinstance(max_score, (int, float)) else str(max_score),
            f"{min_score:.3f}" if isinstance(min_score, (int, float)) else str(min_score),
            str(total_frames),
            f"{suspicious_percentage:.1f}%"
        )
    
    return table

def main():
    """Run comprehensive multi-model testing"""
    
    console.print(Panel.fit(
        "[bold blue]🧪 Comprehensive Analysis Multi-Model Testing[/bold blue]\n"
        "[yellow]Testing all available models with comprehensive analysis endpoint[/yellow]",
        border_style="blue"
    ))
    
    # Check server health
    try:
        headers = {"X-API-Key": API_KEY}
        response = requests.get(f"{BASE_URL}/", headers=headers)
        if response.status_code != 200:
            console.print("[red]❌ Server is not running or not accessible[/red]")
            return
    except Exception as e:
        console.print(f"[red]❌ Server connection failed: {e}[/red]")
        return
    
    # Get available models
    console.print("\n[cyan]🔍 Discovering available models...[/cyan]")
    available_models = get_available_models()
    console.print(f"[green]✅ Found {len(available_models)} models: {', '.join(available_models)}[/green]")
    
    if not available_models:
        console.print("[red]❌ No models available for testing[/red]")
        return
    
    comprehensive_results = []
    individual_results = []
    
    # Test each model with progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        
        for model_name in available_models:
            # Test comprehensive analysis
            task = progress.add_task(f"Testing {model_name} - Comprehensive", total=1)
            console.print(f"\n[yellow]🚀 Testing Comprehensive Analysis: {model_name}[/yellow]")
            
            comp_result = test_comprehensive_analysis_for_model(model_name)
            comprehensive_results.append(comp_result)
            progress.update(task, advance=1)
            
            if comp_result and comp_result.get('success'):
                console.print(f"[green]✅ {model_name} comprehensive analysis completed[/green]")
            else:
                console.print(f"[red]❌ {model_name} comprehensive analysis failed[/red]")
                if comp_result:
                    console.print(f"[red]   Error: {comp_result.get('error', 'Unknown error')}[/red]")
                    console.print(f"[red]   Status Code: {comp_result.get('status_code', 'N/A')}[/red]")
                    if comp_result.get('response_text'):
                        console.print(f"[red]   Response: {comp_result.get('response_text', '')[:200]}...[/red]")
            
            # Test individual endpoints for comparison
            task2 = progress.add_task(f"Testing {model_name} - Individual", total=1)
            console.print(f"[yellow]📊 Testing Individual Endpoints: {model_name}[/yellow]")
            
            individual_result = test_individual_endpoints_for_model(model_name)
            individual_results.append(individual_result)
            progress.update(task2, advance=1)
            
            console.print(f"[green]✅ {model_name} individual endpoints completed[/green]")
    
    # Display results in beautiful tables
    console.print("\n" + "="*80)
    console.print(create_performance_comparison_table(comprehensive_results, individual_results))
    
    console.print("\n")
    console.print(create_processing_breakdown_table(comprehensive_results))
    
    console.print("\n")
    console.print(create_model_metrics_table(comprehensive_results))
    
    # Summary statistics
    successful_tests = [r for r in comprehensive_results if r and r.get('success')]
    if successful_tests:
        avg_efficiency = sum(
            ((next(ir for ir in individual_results if ir['model_name'] == r['model_name'])['total_time'] - r['client_time']) 
             / next(ir for ir in individual_results if ir['model_name'] == r['model_name'])['total_time'] * 100)
            for r in successful_tests
        ) / len(successful_tests)
        
        console.print(Panel.fit(
            f"[bold green]🎉 Testing Summary[/bold green]\n"
            f"[yellow]Models Tested:[/yellow] {len(available_models)}\n"
            f"[yellow]Successful Tests:[/yellow] {len(successful_tests)}\n"
            f"[yellow]Average Efficiency Gain:[/yellow] {avg_efficiency:.1f}%\n"
            f"[yellow]Test Video:[/yellow] {TEST_VIDEO_PATH.name}",
            border_style="green"
        ))
    
    console.print("\n[bold blue]🎯 All tests completed successfully![/bold blue]")

if __name__ == "__main__":
    main()
