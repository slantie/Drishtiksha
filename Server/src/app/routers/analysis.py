# src/app/routers/analysis.py

import os
import asyncio
import logging
from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.responses import StreamingResponse

from src.app.dependencies import get_model_manager, process_video_request
from src.app.schemas import APIResponse, AnalysisData, FramesAnalysisData, ComprehensiveAnalysisData, AudioAnalysisData
from src.app.security import get_api_key
from src.ml.registry import ModelManager

# Use a module-specific logger, consistent with best practices.
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/analyze",
    tags=["Analysis"],
    dependencies=[Depends(get_api_key)]
)

# Common dependency for all routes in this file
VideoProcessingDeps = Depends(process_video_request)

def _log_analysis_result(result_type: str, model_name: str, video_path: str, result: dict):
    """A standardized helper function to log analysis results."""
    header = f"✅ {result_type.upper()} ANALYSIS COMPLETE"
    log_message = [
        "\n" + "=" * 80,
        header,
        "=" * 80,
        f"| 📁 Video: {os.path.basename(video_path)}",
        f"| 🤖 Model: {model_name}",
        f"| 🎯 Prediction: {result.get('prediction', 'N/A')}",
        f"| 📊 Confidence: {result.get('confidence', 0.0):.4f}",
        f"| ⏱️  Processing Time: {result.get('processing_time', 0.0):.2f}s",
    ]
    if result.get('note'):
        log_message.append(f"| 💡 Note: {result.get('note')}")
    log_message.append("=" * 80)
    logger.info("\n".join(log_message))


@router.post("", response_model=APIResponse[AnalysisData])
async def analyze_quick(
    model_manager: ModelManager = Depends(get_model_manager),
    proc_data: tuple = VideoProcessingDeps,
):
    """Performs a comprehensive deepfake analysis with detailed metrics."""
    model_name, video_path = proc_data
    model = model_manager.get_model(model_name)
    
    # REFACTOR: Simplify by always attempting the most detailed prediction.
    # The global exception handler in main.py will catch NotImplementedError gracefully.
    result = await asyncio.to_thread(model.predict_detailed, video_path)
    
    _log_analysis_result("QUICK/DETAILED", model_name, video_path, result)
    
    return APIResponse(model_used=model_name, data=AnalysisData(**result))


@router.post("/frames", response_model=APIResponse[FramesAnalysisData])
async def analyze_frames(
    model_manager: ModelManager = Depends(get_model_manager),
    proc_data: tuple = VideoProcessingDeps,
):
    """Performs frame-by-frame analysis for temporal inspection."""
    model_name, video_path = proc_data
    model = model_manager.get_model(model_name)
    result = await asyncio.to_thread(model.predict_frames, video_path)
    
    _log_analysis_result("FRAMES", model_name, video_path, result)
    
    return APIResponse(model_used=model_name, data=FramesAnalysisData(**result))


@router.post("/visualize")
async def analyze_visual(
    model_manager: ModelManager = Depends(get_model_manager),
    proc_data: tuple = VideoProcessingDeps,
):
    """Generates and streams a video with analysis visualizations."""
    model_name, video_path = proc_data
    model = model_manager.get_model(model_name)
    
    logger.info(f"🎥 Generating visualization for '{os.path.basename(video_path)}' using model '{model_name}'.")
    
    output_video_path = await asyncio.to_thread(model.predict_visual, video_path)
    
    logger.info(f"📤 Streaming visualization video '{os.path.basename(output_video_path)}' to client.")

    async def file_iterator_with_cleanup(path):
        async with open(path, "rb") as f:
            while chunk := await f.read(1024 * 1024): # 1MB chunks
                yield chunk
        try:
            os.remove(path)
            logger.debug(f"Cleaned up temporary visualization file: {path}")
        except OSError as e:
            logger.error(f"Error cleaning up temp file {path}: {e}")

    return StreamingResponse(
        file_iterator_with_cleanup(output_video_path),
        media_type="video/mp4",
        headers={"Content-Disposition": f"attachment; filename=visual_analysis_{model_name}.mp4"}
    )


@router.post("/comprehensive", response_model=APIResponse[ComprehensiveAnalysisData])
async def analyze_comprehensive(
    model_manager: ModelManager = Depends(get_model_manager),
    proc_data: tuple = VideoProcessingDeps,
    video_id: str = Form(None),
    user_id: str = Form(None),
):
    """Performs a comprehensive analysis combining all analysis types in a single request."""
    model_name, video_path = proc_data
    model = model_manager.get_model(model_name)

    # The predict_detailed method now serves as the comprehensive entry point for models that support it.
    result = await asyncio.to_thread(model.predict_detailed, video_path, video_id=video_id, user_id=user_id)
    
    # We will log the main prediction here. Visualization is logged separately if generated.
    _log_analysis_result("COMPREHENSIVE", model_name, video_path, result)
    
    comprehensive_data = ComprehensiveAnalysisData(**result)
    
    return APIResponse(model_used=model_name, data=comprehensive_data)


@router.get("/visualization/{filename}")
async def download_visualization(filename: str):
    """Downloads a previously generated visualization video by filename."""
    import tempfile
    
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, filename)
    
    if not os.path.exists(video_path):
        logger.warning(f"Visualization download request for non-existent file: {filename}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Visualization file '{filename}' not found or has expired."
        )
    
    logger.info(f"📥 Serving download for visualization file: {filename}")

    async def file_iterator_with_cleanup(path):
        async with open(path, "rb") as f:
            while chunk := await f.read(1024 * 1024):
                yield chunk
        try:
            os.remove(path)
            logger.debug(f"Cleaned up temporary visualization file after download: {path}")
        except OSError as e:
            logger.error(f"Error cleaning up temp file {path} after download: {e}")

    return StreamingResponse(
        file_iterator_with_cleanup(video_path),
        media_type="video/mp4",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.post("/audio", response_model=APIResponse[AudioAnalysisData])
async def analyze_audio(
    model_manager: ModelManager = Depends(get_model_manager),
    proc_data: tuple = VideoProcessingDeps,
):
    """Performs a comprehensive analysis on the audio track of a video."""
    model_name, video_path = proc_data
    
    model = model_manager.get_model(model_name)
    if not getattr(model.config, 'isAudio', False):
         raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{model_name}' does not support audio analysis. Please use an audio-specific model.",
        )

    result = await asyncio.to_thread(model.predict_detailed, video_path)

    _log_analysis_result("AUDIO", model_name, video_path, result)

    return APIResponse(model_used=model_name, data=AudioAnalysisData(**result))