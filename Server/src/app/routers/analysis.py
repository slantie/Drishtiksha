# src/app/routers/analysis.py

import os
import asyncio
import aiofiles
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from src.app.dependencies import get_model_manager, process_video_request
from src.app.schemas import (
    APIResponse, QuickAnalysisData, DetailedAnalysisData, FramesAnalysisData
)
from src.app.security import get_api_key
from src.ml.registry import ModelManager

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
    """Performs a quick, high-level deepfake analysis."""
    model_name, video_path = proc_data
    model = model_manager.get_model(model_name)
    
    # Run the model inference in a separate thread to avoid blocking the event loop
    result = await asyncio.to_thread(model.predict, video_path)
    
    return APIResponse(model_used=model_name, data=QuickAnalysisData(**result))

@router.post("/detailed", response_model=APIResponse)
async def analyze_detailed(
    model_manager: ModelManager = Depends(get_model_manager),
    proc_data: tuple = VideoProcessingDeps,
):
    """Provides a detailed, metric-rich deepfake analysis."""
    model_name, video_path = proc_data
    model = model_manager.get_model(model_name)
    result = await asyncio.to_thread(model.predict_detailed, video_path)
    return APIResponse(model_used=model_name, data=DetailedAnalysisData(**result))

@router.post("/frames", response_model=APIResponse)
async def analyze_frames(
    model_manager: ModelManager = Depends(get_model_manager),
    proc_data: tuple = VideoProcessingDeps,
):
    """Performs frame-by-frame analysis for temporal inspection."""
    model_name, video_path = proc_data
    model = model_manager.get_model(model_name)
    result = await asyncio.to_thread(model.predict_frames, video_path)
    return APIResponse(model_used=model_name, data=FramesAnalysisData(**result))


@router.post("/visualize")
async def analyze_visual(
    model_manager: ModelManager = Depends(get_model_manager),
    proc_data: tuple = VideoProcessingDeps,
):
    """Generates and streams a video with analysis visualizations."""
    model_name, video_path = proc_data
    model = model_manager.get_model(model_name)
    
    try:
        # The model method returns the path to the generated video
        output_video_path = await asyncio.to_thread(model.predict_visual, video_path)
    except NotImplementedError:
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