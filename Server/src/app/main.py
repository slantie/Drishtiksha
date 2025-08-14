# src/app/main.py

import os
import uuid
import tempfile
import aiofiles
from typing import AsyncGenerator
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from src.config import settings
from src.ml.registry import ModelManager
from src.app.schemas import HealthResponse, ModelInfoResponse, AnalysisResponse, AnalysisResult
from src.app.security import get_api_key

# Global state to hold the model manager
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the default ML model on startup and clean up on shutdown."""
    print(f"🚀 Starting {settings.project_name}...")
    app_state["model_manager"] = ModelManager(settings.models)
    try:
        # Pre-load the default model to ensure it's ready and handle any loading errors on startup
        print(f"Pre-loading default model: '{settings.default_model_name.upper()}'...")
        app_state["model_manager"].get_model(settings.default_model_name)
        yield
    finally:
        print("🔌 Shutting down server...")
        app_state.clear()

app = FastAPI(title=settings.project_name, version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/health", response_model=HealthResponse, tags=["Status"])
def health_check():
    """Performs a health check on the server and model."""
    manager = app_state.get("model_manager")
    is_model_loaded = manager is not None and settings.default_model_name in manager._models
    return {"status": "healthy", "model_loaded": is_model_loaded, "default_model": settings.default_model_name}

@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    """Provides metadata about the currently loaded default model."""
    manager = app_state["model_manager"]
    model = manager.get_model(settings.default_model_name)
    return {"success": True, "model_info": model.get_info()}

@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"], dependencies=[Depends(get_api_key)])
async def analyze_video(
    video: UploadFile = File(..., description="The video file to analyze."),
    video_id: str = Form(default_factory=lambda: str(uuid.uuid4()), description="Optional unique ID for the video.")
):
    """Analyzes a video to detect if it's a deepfake."""
    manager = app_state.get("model_manager")
    model = manager.get_model(settings.default_model_name)
    
    # Use a temporary file to securely handle the upload
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.filename)[1]) as tmp:
            tmp.write(await video.read())
            temp_path = tmp.name
        
        result_dict = model.predict(temp_path)
        result_dict["model_version"] = settings.default_model_name
        
        return AnalysisResponse(
            success=True,
            video_id=video_id,
            result=AnalysisResult(**result_dict)
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:              
        print(f"An unexpected error occurred during analysis: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Analysis failed: {e}")
    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

def cleanup_files(*paths: str):
    """Deletes all provided file paths."""
    for path in paths:
        if path and os.path.exists(path):
            os.remove(path)
            print(f"Cleaned up temporary file: {path}")

async def file_iterator(file_path: str) -> AsyncGenerator[bytes, None]:
    """Asynchronously reads a file in chunks."""
    async with aiofiles.open(file_path, "rb") as f:
        while chunk := await f.read(1024 * 1024):
            yield chunk

@app.post("/analyze/visualize", tags=["Analysis"], dependencies=[Depends(get_api_key)])
async def analyze_video_visualized(
    video: UploadFile = File(..., description="The video file to generate a visualized analysis for.")
):
    """
    Analyzes a video and returns a new video file with an overlaid graph.
    """
    manager = app_state.get("model_manager")
    model = manager.get_model(settings.default_model_name)
    
    input_temp_path = None
    output_video_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.filename)[1]) as tmp:
            tmp.write(await video.read())
            input_temp_path = tmp.name
        
        output_video_path = model.predict_visualized(input_temp_path)
        
        # Define the cleanup task for BOTH files
        cleanup_task = BackgroundTask(cleanup_files, input_temp_path, output_video_path)
        
        # The file_iterator only needs the path. The cleanup is handled by the `background` param.
        return StreamingResponse(
            file_iterator(output_video_path),
            media_type="video/mp4",
            headers={"Content-Disposition": f"attachment; filename=visual_analysis_{video.filename}"},
            background=cleanup_task
        )

    except Exception as e:
        # If an error happens before we return the response, clean up manually.
        cleanup_files(input_temp_path, output_video_path)
        print(f"An unexpected error occurred during visual analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Visual analysis failed: {e}")