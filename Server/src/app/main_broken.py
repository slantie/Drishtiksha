# src/app/main.py

import os
import uuid
import tempfile
import aiofiles
import asyncio
import threading
from typing import AsyncGenerator
from contextlib import asynccontextmanager
from collections import defaultdict
from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from src.config import settings
from src.ml.registry import ModelManager
from src.app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    AnalysisResponse,
    AnalysisResult,
    DetailedAnalysisResponse,
    DetailedAnalysisResult,
    FrameAnalysisResponse,
    FrameAnalysisSummary,
)
from src.app.security import get_api_key

# Global state to hold the model manager and load balancing
app_state = {}

# Load balancing and request tracking
request_counts = defaultdict(int)  # Track requests per model
request_semaphores = defaultdict(lambda: asyncio.Semaphore(2))  # Max 2 concurrent requests per model
request_lock = threading.Lock()

def get_optimal_model(requested_model: str = None) -> str:
    """
    Load balancing logic to select the optimal model.
    If a specific model is requested, use it. Otherwise, select based on current load.
    """
    if requested_model:
        return requested_model
    
    # If no specific model requested, use load balancing
    with request_lock:
        available_models = list(settings.models.keys())
        if not available_models:
            return settings.default_model_name
        
        # Select model with lowest current request count
        optimal_model = min(available_models, key=lambda m: request_counts[m])
        return optimal_model

async def safe_model_execution(model, method_name: str, *args, **kwargs):
    """
    Safely execute model methods with proper error handling and fallbacks.
    """
    try:
        # Check if method exists
        if not hasattr(model, method_name):
            return {
                "error": f"Model does not support {method_name} method",
                "supported_methods": [m for m in dir(model) if not m.startswith('_') and callable(getattr(model, m))]
            }
        
        method = getattr(model, method_name)
        result = method(*args, **kwargs)
        return {"success": True, "result": result}
        
    except Exception as e:
        error_msg = f"Model execution failed: {str(e)}"
        print(f"ERROR in {method_name}: {error_msg}")
        return {
            "error": error_msg,
            "method": method_name,
            "fallback_available": False
        }


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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Status"])
def health_check():
    """Performs a basic health check on the server."""
    try:
        manager = app_state.get("model_manager")
        is_model_loaded = (
            manager is not None and settings.default_model_name in manager._models
        )
        return {
            "status": "healthy",
            "model_loaded": is_model_loaded,
            "default_model": settings.default_model_name,
        }
    except Exception as e:
        print(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "default_model": settings.default_model_name,
        }


@app.get("/ping", tags=["Status"])
def ping():
    """Simple ping endpoint for basic connectivity check."""
    return {"status": "ok", "message": "Server is running"}


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    """Provides metadata about the currently loaded default model."""
    manager = app_state["model_manager"]
    model = manager.get_model(settings.default_model_name)
    return {"success": True, "model_info": model.get_info()}


@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    tags=["Analysis"],
    dependencies=[Depends(get_api_key)],
)
async def analyze_video(
    video: UploadFile = File(..., description="The video file to analyze."),
    video_id: str = Form(
        default_factory=lambda: str(uuid.uuid4()),
        description="Optional unique ID for the video.",
    ),
    model_name: str = Form(
        default="", description="Optional model name (defaults to load-balanced selection)."
    ),
):
    """Analyzes a video to detect if it's a deepfake with load balancing and timeout handling."""
    request_id = str(uuid.uuid4())[:8]
    print(f"[{request_id}] Starting analysis for video_id: {video_id}")

    manager = app_state.get("model_manager")
    
    # Use load balancing to select optimal model
    target_model_name = get_optimal_model(model_name)
    
    # Apply semaphore for load balancing
    async with request_semaphores[target_model_name]:
        with request_lock:
            request_counts[target_model_name] += 1
        
        try:
            model = manager.get_model(target_model_name)
            print(f"[{request_id}] Using model: {target_model_name}")

            # Use a temporary file to securely handle the upload
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(video.filename)[1]
                ) as tmp:
                    content = await video.read()
                    tmp.write(content)
                    temp_path = tmp.name

                print(f"[{request_id}] Saved video to temp file: {temp_path}")

                # Safe model execution with error handling
                execution_result = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: safe_model_execution(model, "predict", temp_path, 30)
                )
                
                if "error" in execution_result:
                    # Graceful error handling - provide detailed error info
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail={
                            "error": execution_result["error"],
                            "model_used": target_model_name,
                            "method": "predict",
                            "supported_methods": execution_result.get("supported_methods", []),
                            "fallback_suggestion": "Try using a different model or check video format"
                        }
                    )

                result_dict = execution_result["result"]
                result_dict["model_version"] = target_model_name

                return AnalysisResponse(
                    success=True,
                    video_id=video_id,
                    result=AnalysisResult(**result_dict),
                )

            except HTTPException:
                raise
            except Exception as e:
                print(f"[{request_id}] Analysis failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": f"Analysis failed: {str(e)}",
                        "model_used": target_model_name,
                        "request_id": request_id
                    }
                )
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        finally:
            # Decrement request count
            with request_lock:
                request_counts[target_model_name] = max(0, request_counts[target_model_name] - 1)


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
    video: UploadFile = File(
        ..., description="The video file to generate a visualized analysis for."
    ),
    model_name: str = Form(
        default="", description="Optional model name (defaults to load-balanced selection)."
    ),
):
    """
    Analyzes a video and returns a new video file with an overlaid graph.
    Enhanced with load balancing and graceful error handling.
    """
    manager = app_state.get("model_manager")
    request_id = str(uuid.uuid4())[:8]

    # Use load balancing to select optimal model
    target_model_name = get_optimal_model(model_name)
    
    # Apply semaphore for load balancing
    async with request_semaphores[target_model_name]:
        with request_lock:
            request_counts[target_model_name] += 1
        
        try:
            model = manager.get_model(target_model_name)
            print(f"[{request_id}] Starting visualization with model: {target_model_name}")

            # Check if model supports visualization
            if not hasattr(model, "predict_visualized"):
                return {
                    "success": False,
                    "error": f"Model '{target_model_name}' does not support visualization.",
                    "supported_methods": [m for m in dir(model) if not m.startswith('_') and callable(getattr(model, m))],
                    "suggestion": "Try using SIGLIP-LSTM-V3 or ColorCues models for visualization"
                }

            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(video.filename)[1]
                ) as tmp:
                    tmp.write(await video.read())
                    temp_path = tmp.name

                # Safe model execution
                execution_result = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: safe_model_execution(model, "predict_visualized", temp_path)
                )
                
                if "error" in execution_result:
                    return {
                        "success": False,
                        "error": execution_result["error"],
                        "model_used": target_model_name,
                        "supported_methods": execution_result.get("supported_methods", [])
                    }

                visualization_path = execution_result["result"]

                return StreamingResponse(
                    file_iterator(visualization_path),
                    media_type="video/mp4",
                    headers={"Content-Disposition": "attachment; filename=analysis_visualization.mp4"},
                    background=BackgroundTask(cleanup_files, temp_path, visualization_path),
                )

            except Exception as e:
                print(f"[{request_id}] Visualization failed: {e}")
                return {
                    "success": False,
                    "error": f"Visualization generation failed: {str(e)}",
                    "model_used": target_model_name,
                    "request_id": request_id
                }
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        finally:
            # Decrement request count
            with request_lock:
                request_counts[target_model_name] = max(0, request_counts[target_model_name] - 1)


@app.post(
    "/analyze/detailed",
    response_model=DetailedAnalysisResponse,
    tags=["Analysis"],
    dependencies=[Depends(get_api_key)],
)
async def analyze_video_detailed(
    video: UploadFile = File(
        ..., description="The video file to analyze with detailed metrics."
    ),
    video_id: str = Form(
        default_factory=lambda: str(uuid.uuid4()),
        description="Optional unique ID for the video.",
    ),
    model_name: str = Form(
        default="", description="Optional model name (defaults to load-balanced selection)."
    ),
):
    """Analyzes a video with detailed frame-by-frame metrics with load balancing."""
    request_id = str(uuid.uuid4())[:8]
    manager = app_state.get("model_manager")

    # Use load balancing to select optimal model
    target_model_name = get_optimal_model(model_name)
    
    # Apply semaphore for load balancing
    async with request_semaphores[target_model_name]:
        with request_lock:
            request_counts[target_model_name] += 1
        
        try:
            model = manager.get_model(target_model_name)
            print(f"[{request_id}] Starting detailed analysis with model: {target_model_name}")

            # Check if model supports detailed analysis
            if not hasattr(model, "predict_with_metrics"):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "error": f"Model '{target_model_name}' does not support detailed analysis.",
                        "supported_methods": [m for m in dir(model) if not m.startswith('_') and callable(getattr(model, m))],
                        "suggestion": "Use SIGLIP-LSTM-V3 or ColorCues models for detailed analysis"
                    }
                )

            # Use a temporary file to securely handle the upload
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(video.filename)[1]
                ) as tmp:
                    tmp.write(await video.read())
                    temp_path = tmp.name

                # Safe model execution
                execution_result = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: safe_model_execution(model, "predict_with_metrics", temp_path)
                )
                
                if "error" in execution_result:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail={
                            "error": execution_result["error"],
                            "model_used": target_model_name,
                            "supported_methods": execution_result.get("supported_methods", [])
                        }
                    )

                result_dict = execution_result["result"]
                result_dict["model_version"] = target_model_name

                return DetailedAnalysisResponse(
                    success=True,
                    video_id=video_id,
                    result=DetailedAnalysisResult(**result_dict),
                )
            except HTTPException:
                raise
            except Exception as e:
                print(f"[{request_id}] Detailed analysis failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": f"Detailed analysis failed: {str(e)}",
                        "model_used": target_model_name,
                        "request_id": request_id
                    }
                )
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        finally:
            # Decrement request count
            with request_lock:
                request_counts[target_model_name] = max(0, request_counts[target_model_name] - 1)


@app.post(
):
    """Analyzes a video with detailed frame-by-frame metrics. Requires LSTMDetectorV3."""
    manager = app_state.get("model_manager")

    # Use specified model or default
    target_model_name = model_name if model_name else settings.default_model_name
    model = manager.get_model(target_model_name)

    # Check if model supports detailed analysis
    if not hasattr(model, "predict_with_metrics"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{target_model_name}' does not support detailed analysis. Use LSTMDetectorV3.",
        )

    # Use a temporary file to securely handle the upload
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(video.filename)[1]
        ) as tmp:
            tmp.write(await video.read())
            temp_path = tmp.name

        result_dict = model.predict_with_metrics(temp_path)
        result_dict["model_version"] = target_model_name

        return DetailedAnalysisResponse(
            success=True,
            video_id=video_id,
            result=DetailedAnalysisResult(**result_dict),
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        print(f"An unexpected error occurred during detailed analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detailed analysis failed: {e}",
        )
    finally:
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post(
    "/analyze/frames",
    response_model=FrameAnalysisResponse,
    tags=["Analysis"],
    dependencies=[Depends(get_api_key)],
)
async def analyze_video_frames(
    video: UploadFile = File(
        ..., description="The video file to analyze frame by frame."
    ),
    video_id: str = Form(
        default_factory=lambda: str(uuid.uuid4()),
        description="Optional unique ID for the video.",
    ),
    model_name: str = Form(
        default="", description="Optional model name (defaults to server default)."
    ),
):
    """Quick frame analysis summary without visualization. Requires LSTMDetectorV3."""
    manager = app_state.get("model_manager")

    # Use specified model or default
    target_model_name = model_name if model_name else settings.default_model_name
    model = manager.get_model(target_model_name)

    # Check if model supports frame analysis
    if not hasattr(model, "get_frame_analysis_summary"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{target_model_name}' does not support frame analysis. Use LSTMDetectorV3.",
        )

    # Use a temporary file to securely handle the upload
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(video.filename)[1]
        ) as tmp:
            tmp.write(await video.read())
            temp_path = tmp.name

        summary_dict = model.get_frame_analysis_summary(temp_path)

        if "error" in summary_dict:
            raise ValueError(summary_dict["error"])

        return FrameAnalysisResponse(
            success=True,
            video_id=video_id,
            summary=FrameAnalysisSummary(**summary_dict),
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        print(f"An unexpected error occurred during frame analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Frame analysis failed: {e}",
        )
    finally:
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
