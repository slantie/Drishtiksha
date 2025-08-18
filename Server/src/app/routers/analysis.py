# src/app/routers/analysis.py

import os
import asyncio
import aiofiles
import logging
from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from src.app.dependencies import get_model_manager, process_video_request
from src.app.schemas import (
    APIResponse, AnalysisData, QuickAnalysisData, DetailedAnalysisData, FramesAnalysisData, ComprehensiveAnalysisData
)
from src.app.security import get_api_key
from src.ml.registry import ModelManager

# Set up logger for terminal output
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/analyze",
    tags=["Analysis"],
    dependencies=[Depends(get_api_key)]
)

# Common dependency for all routes in this file
VideoProcessingDeps = Depends(process_video_request)

@router.post("", response_model=APIResponse)
async def analyze_quick(
    model_manager: ModelManager = Depends(get_model_manager),
    proc_data: tuple = VideoProcessingDeps,
):
    """Performs a comprehensive deepfake analysis with detailed metrics."""
    model_name, video_path = proc_data
    model = model_manager.get_model(model_name)
    
    # Try detailed analysis first, fallback to basic predict if not available
    try:
        result = await asyncio.to_thread(model.predict_detailed, video_path)
        analysis_type = "COMPREHENSIVE ANALYSIS"
    except NotImplementedError:
        result = await asyncio.to_thread(model.predict, video_path)
        analysis_type = "QUICK ANALYSIS"
    
    # Print the model response to terminal
    print("=" * 80)
    print(f"🔍 {analysis_type} RESPONSE")
    print("=" * 80)
    print(f"📁 Video: {os.path.basename(video_path)}")
    print(f"🤖 Model: {model_name}")
    print(f"🎯 Prediction: {result.get('prediction', 'N/A')}")
    print(f"📊 Confidence: {result.get('confidence', 'N/A'):.3f}" if isinstance(result.get('confidence'), (int, float)) else f"📊 Confidence: {result.get('confidence', 'N/A')}")
    print(f"⏱️  Processing Time: {result.get('processing_time', 'N/A'):.3f}s" if isinstance(result.get('processing_time'), (int, float)) else f"⏱️  Processing Time: {result.get('processing_time', 'N/A')}")
    
    # Show detailed metrics if available
    metrics = result.get('metrics', {})
    if metrics:
        # Handle different metric structures for different models
        frame_count = metrics.get('frame_count') or metrics.get('sequence_count', 'N/A')
        avg_score = metrics.get('final_average_score', 'N/A')
        max_score = metrics.get('max_score', 'N/A')
        min_score = metrics.get('min_score', 'N/A')
        suspicious_count = metrics.get('suspicious_frames_count') or metrics.get('suspicious_sequences_count', 'N/A')
        
        print(f"📈 Analysis Units: {frame_count}")
        print(f"📊 Average Score: {avg_score:.3f}" if isinstance(avg_score, (int, float)) else f"📊 Average Score: {avg_score}")
        print(f"📈 Max Score: {max_score:.3f}" if isinstance(max_score, (int, float)) else f"📈 Max Score: {max_score}")
        print(f"📉 Min Score: {min_score:.3f}" if isinstance(min_score, (int, float)) else f"📉 Min Score: {min_score}")
        print(f"🔍 Suspicious Units: {suspicious_count}")
    
    # Show note if available
    if result.get('note'):
        print(f"💡 Note: {result.get('note')}")
    
    print("=" * 80)
    
    # Return unified response
    return APIResponse(model_used=model_name, data=AnalysisData(**result))

@router.post("/frames", response_model=APIResponse)
async def analyze_frames(
    model_manager: ModelManager = Depends(get_model_manager),
    proc_data: tuple = VideoProcessingDeps,
):
    """Performs frame-by-frame analysis for temporal inspection."""
    model_name, video_path = proc_data
    model = model_manager.get_model(model_name)
    result = await asyncio.to_thread(model.predict_frames, video_path)
    
    # Print the frames analysis response to terminal
    print("=" * 80)
    print("🎬 FRAMES ANALYSIS RESPONSE")
    print("=" * 80)
    print(f"📁 Video: {os.path.basename(video_path)}")
    print(f"🤖 Model: {model_name}")
    print(f"🎯 Overall Prediction: {result.get('overall_prediction', 'N/A')}")
    print(f"📊 Overall Confidence: {result.get('overall_confidence', 'N/A'):.3f}" if isinstance(result.get('overall_confidence'), (int, float)) else f"📊 Overall Confidence: {result.get('overall_confidence', 'N/A')}")
    print(f"⏱️  Processing Time: {result.get('processing_time', 'N/A'):.3f}s" if isinstance(result.get('processing_time'), (int, float)) else f"⏱️  Processing Time: {result.get('processing_time', 'N/A')}")
    
    # Show frame-level information
    frame_predictions = result.get('frame_predictions', [])
    if frame_predictions:
        print(f"🎬 Total Frames Analyzed: {len(frame_predictions)}")
        fake_frames = sum(1 for frame in frame_predictions if frame.get('prediction') == 'FAKE')
        print(f"🚨 Suspicious Frames: {fake_frames} ({fake_frames/len(frame_predictions)*100:.1f}%)")
        
        # Show temporal analysis if available
        temporal = result.get('temporal_analysis', {})
        if temporal:
            print(f"📈 Consistency Score: {temporal.get('consistency_score', 'N/A'):.3f}" if isinstance(temporal.get('consistency_score'), (int, float)) else f"📈 Consistency Score: {temporal.get('consistency_score', 'N/A')}")
    print("=" * 80)
    
    return APIResponse(model_used=model_name, data=FramesAnalysisData(**result))


@router.post("/visualize")
async def analyze_visual(
    model_manager: ModelManager = Depends(get_model_manager),
    proc_data: tuple = VideoProcessingDeps,
):
    """Generates and streams a video with analysis visualizations."""
    model_name, video_path = proc_data
    model = model_manager.get_model(model_name)
    
    # Print the visual analysis request to terminal
    print("=" * 80)
    print("🎥 VISUAL ANALYSIS REQUEST")
    print("=" * 80)
    print(f"📁 Video: {os.path.basename(video_path)}")
    print(f"🤖 Model: {model_name}")
    print("🎬 Generating visualization video...")
    print("=" * 80)
    
    try:
        # The model method returns the path to the generated video
        output_video_path = await asyncio.to_thread(model.predict_visual, video_path)
        
        # Print success message
        print("=" * 80)
        print("✅ VISUAL ANALYSIS COMPLETED")
        print("=" * 80)
        print(f"📁 Input Video: {os.path.basename(video_path)}")
        print(f"🤖 Model: {model_name}")
        print(f"🎬 Output Video: {os.path.basename(output_video_path)}")
        print("📤 Streaming visualization video to client...")
        print("=" * 80)
        
    except NotImplementedError:
        print("=" * 80)
        print("❌ VISUAL ANALYSIS ERROR")
        print("=" * 80)
        print(f"🤖 Model: {model_name}")
        print("⚠️  Visual analysis not supported by this model")
        print("=" * 80)
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"Model '{model_name}' does not support visual analysis."
        )

    # This helper will stream the file and then delete it.
    async def file_iterator_with_cleanup(path):
        async with aiofiles.open(path, "rb") as f:
            while chunk := await f.read(1024 * 1024):
                yield chunk
        os.remove(path)

    return StreamingResponse(
        file_iterator_with_cleanup(output_video_path),
        media_type="video/mp4",
        headers={"Content-Disposition": f"attachment; filename=visual_analysis_{model_name}.mp4"}
    )


@router.post("/comprehensive", response_model=APIResponse)
async def analyze_comprehensive(
    model_manager: ModelManager = Depends(get_model_manager),
    proc_data: tuple = VideoProcessingDeps,
    include_frames: bool = Form(True),
    include_visualization: bool = Form(True),
    video_id: str = Form(None),
    user_id: str = Form(None),
):
    """
    Performs a comprehensive analysis combining all analysis types in a single request.
    This reduces computation time by reusing extracted frames and processing.
    
    Args:
        include_frames: Whether to include frame-by-frame analysis
        include_visualization: Whether to generate visualization video
    """
    import time
    
    model_name, video_path = proc_data
    model = model_manager.get_model(model_name)
    
    print("=" * 80)
    print("🚀 COMPREHENSIVE ANALYSIS STARTED")
    print("=" * 80)
    print(f"📁 Video: {os.path.basename(video_path)}")
    print(f"🤖 Model: {model_name}")
    print(f"🎬 Include Frames: {include_frames}")
    print(f"📊 Include Visualization: {include_visualization}")
    print("⏳ Processing all analysis types...")
    print("=" * 80)
    
    start_time = time.time()
    processing_breakdown = {}
    
    # 1. Basic comprehensive analysis (detailed if available)
    analysis_start = time.time()
    try:
        # Attempt to call the full-featured detailed method first
        basic_result = await asyncio.to_thread(model.predict_detailed, video_path, video_id=video_id, user_id=user_id)
        analysis_type = "COMPREHENSIVE"
    except (NotImplementedError, TypeError):
        logger.warning(f"Model '{model_name}' does not support full detailed analysis. Falling back to basic prediction.")
        basic_result = await asyncio.to_thread(model.predict, video_path)
        analysis_type = "QUICK (Fallback)"
    processing_breakdown["basic_analysis"] = time.time() - analysis_start

    # 2. Frame-by-frame analysis (if requested)
    frames_result = None
    if include_frames:
        frames_start = time.time()
        try:
            frames_result = await asyncio.to_thread(model.predict_frames, video_path)
        except NotImplementedError:
            print("⚠️  Frame analysis not supported by this model")
        processing_breakdown["frames_analysis"] = time.time() - frames_start
    
    # 3. Visualization (if requested)
    visualization_filename = None
    visualization_generated = False
    if include_visualization:
        viz_start = time.time()
        try:
            output_video_path = await asyncio.to_thread(model.predict_visual, video_path)
            visualization_filename = os.path.basename(output_video_path)
            visualization_generated = True
        except NotImplementedError:
            print("⚠️  Visual analysis not supported by this model")
        processing_breakdown["visualization"] = time.time() - viz_start
    
    total_processing_time = time.time() - start_time
    processing_breakdown["total"] = total_processing_time
    
    # Print comprehensive results
    print("=" * 80)
    print(f"✅ COMPREHENSIVE ANALYSIS COMPLETED")
    print("=" * 80)
    print(f"📁 Video: {os.path.basename(video_path)}")
    print(f"🤖 Model: {model_name}")
    print(f"🎯 Analysis Type: {analysis_type}")
    print(f"🎯 Prediction: {basic_result.get('prediction', 'N/A')}")
    print(f"📊 Confidence: {basic_result.get('confidence', 'N/A'):.3f}" if isinstance(basic_result.get('confidence'), (int, float)) else f"📊 Confidence: {basic_result.get('confidence', 'N/A')}")
    print(f"⏱️  Total Processing Time: {total_processing_time:.3f}s")
    
    # Processing breakdown
    print("\n📊 Processing Breakdown:")
    for step, duration in processing_breakdown.items():
        if step != "total":
            print(f"   {step.replace('_', ' ').title()}: {duration:.3f}s")
    
    # Show detailed metrics if available
    metrics = basic_result.get('metrics', {})
    if metrics:
        frame_count = metrics.get('frame_count') or metrics.get('sequence_count', 'N/A')
        avg_score = metrics.get('final_average_score', 'N/A')
        max_score = metrics.get('max_score', 'N/A')
        min_score = metrics.get('min_score', 'N/A')
        suspicious_count = metrics.get('suspicious_frames_count') or metrics.get('suspicious_sequences_count', 'N/A')
        
        print(f"\n📈 Analysis Metrics:")
        print(f"   Analysis Units: {frame_count}")
        print(f"   Average Score: {avg_score:.3f}" if isinstance(avg_score, (int, float)) else f"   Average Score: {avg_score}")
        print(f"   Max Score: {max_score:.3f}" if isinstance(max_score, (int, float)) else f"   Max Score: {max_score}")
        print(f"   Min Score: {min_score:.3f}" if isinstance(min_score, (int, float)) else f"   Min Score: {min_score}")
        print(f"   Suspicious Units: {suspicious_count}")
    
    # Frame analysis summary
    if frames_result:
        frame_predictions = frames_result.get('frame_predictions', [])
        if frame_predictions:
            fake_frames = sum(1 for frame in frame_predictions if frame.get('prediction') == 'FAKE')
            print(f"\n🎬 Frame Analysis:")
            print(f"   Total Frames: {len(frame_predictions)}")
            print(f"   Suspicious Frames: {fake_frames} ({fake_frames/len(frame_predictions)*100:.1f}%)")
    
    # Visualization status
    if include_visualization:
        print(f"\n📊 Visualization:")
        if visualization_generated:
            print(f"   Status: ✅ Generated")
            print(f"   File: {visualization_filename}")
        else:
            print(f"   Status: ❌ Not supported by model")
    
    print("=" * 80)
    
    # Prepare comprehensive response
    comprehensive_data = ComprehensiveAnalysisData(
        **basic_result,
        frames_analysis=FramesAnalysisData(**frames_result) if frames_result else None,
        visualization_generated=visualization_generated,
        visualization_filename=visualization_filename,
        processing_breakdown=processing_breakdown
    )
    
    return APIResponse(model_used=model_name, data=comprehensive_data)


@router.get("/visualization/{filename}")
async def download_visualization(filename: str):
    """
    Downloads a previously generated visualization video by filename.
    This is used in conjunction with the comprehensive analysis endpoint.
    """
    import tempfile
    
    # Construct the full path to the temporary file
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, filename)
    
    if not os.path.exists(video_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Visualization file '{filename}' not found or has expired."
        )
    
    print("=" * 80)
    print("📥 VISUALIZATION DOWNLOAD")
    print("=" * 80)
    print(f"📁 File: {filename}")
    print("📤 Streaming visualization video to client...")
    print("=" * 80)

    # Stream the file and delete after download
    async def file_iterator_with_cleanup(path):
        async with aiofiles.open(path, "rb") as f:
            while chunk := await f.read(1024 * 1024):
                yield chunk
        os.remove(path)

    return StreamingResponse(
        file_iterator_with_cleanup(video_path),
        media_type="video/mp4",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )