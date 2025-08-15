# src/app/main_enhanced.py - Enhanced with load balancing and comprehensive endpoint support

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
request_semaphores = defaultdict(
    lambda: asyncio.Semaphore(2)
)  # Max 2 concurrent requests per model
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


def safe_model_execution(model, method_name: str, *args, **kwargs):
    """
    Safely execute model methods with proper error handling and fallbacks.
    Handles different method signatures gracefully.
    """
    try:
        # Check if method exists
        if not hasattr(model, method_name):
            return {
                "error": f"Model does not support {method_name} method",
                "supported_methods": [
                    m
                    for m in dir(model)
                    if not m.startswith("_") and callable(getattr(model, m))
                ],
            }

        method = getattr(model, method_name)

        # Handle different method signatures gracefully
        if method_name == "predict":
            # Try with num_frames parameter first (V3), fallback to video_path only (V2)
            try:
                if len(args) >= 2:  # video_path + num_frames
                    result = method(args[0], args[1])
                else:
                    result = method(args[0])
            except TypeError as te:
                # If method signature doesn't match, try with just video_path
                if "takes" in str(te) and "positional arguments" in str(te):
                    result = method(args[0])  # Just video_path
                else:
                    raise te
        else:
            # For other methods, use all provided arguments
            result = method(*args, **kwargs)

        return {"success": True, "result": result}

    except Exception as e:
        error_msg = f"Model execution failed: {str(e)}"
        print(f"ERROR in {method_name}: {error_msg}")
        return {"error": error_msg, "method": method_name, "fallback_available": False}


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


# Health and Status Endpoints
@app.get("/health", response_model=HealthResponse, tags=["Status"])
async def health_check():
    """Basic health check endpoint."""
    manager = app_state.get("model_manager")
    if not manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not available",
        )

    default_model = manager.get_model(settings.default_model_name)
    model_info = default_model.get_info()

    return HealthResponse(
        status="ok",
        model_loaded=True,
        default_model=model_info.get("model_name", settings.default_model_name),
    )


@app.get("/ping", tags=["Status"])
async def ping():
    """Simple ping endpoint."""
    return {"message": "pong"}


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get information about the currently loaded model."""
    manager = app_state["model_manager"]
    model = manager.get_model(settings.default_model_name)
    return {"success": True, "model_info": model.get_info()}


# Analysis Endpoints with Load Balancing


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
        default="",
        description="Optional model name (defaults to load-balanced selection).",
    ),
):
    """Analyzes a video to detect if it's a deepfake with load balancing and enhanced error handling."""
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
                    None, lambda: safe_model_execution(model, "predict", temp_path, 30)
                )

                if "error" in execution_result:
                    # Graceful error handling - provide detailed error info
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail={
                            "error": execution_result["error"],
                            "model_used": target_model_name,
                            "method": "predict",
                            "supported_methods": execution_result.get(
                                "supported_methods", []
                            ),
                            "fallback_suggestion": "Try using a different model or check video format",
                        },
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
                        "request_id": request_id,
                    },
                )
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

        finally:
            # Decrement request count
            with request_lock:
                request_counts[target_model_name] = max(
                    0, request_counts[target_model_name] - 1
                )


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
        default="",
        description="Optional model name (defaults to load-balanced selection).",
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
            print(
                f"[{request_id}] Starting detailed analysis with model: {target_model_name}"
            )

            # Check if model supports detailed analysis with graceful error
            if not hasattr(model, "predict_with_metrics"):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "error": f"Model '{target_model_name}' does not support detailed analysis.",
                        "supported_methods": [
                            m
                            for m in dir(model)
                            if not m.startswith("_") and callable(getattr(model, m))
                        ],
                        "suggestion": "Use SIGLIP-LSTM-V3 or ColorCues models for detailed analysis",
                        "available_models": list(settings.models.keys()),
                    },
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
                    lambda: safe_model_execution(
                        model, "predict_with_metrics", temp_path
                    ),
                )

                if "error" in execution_result:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail={
                            "error": execution_result["error"],
                            "model_used": target_model_name,
                            "supported_methods": execution_result.get(
                                "supported_methods", []
                            ),
                        },
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
                        "request_id": request_id,
                    },
                )
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

        finally:
            # Decrement request count
            with request_lock:
                request_counts[target_model_name] = max(
                    0, request_counts[target_model_name] - 1
                )


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
        default="",
        description="Optional model name (defaults to load-balanced selection).",
    ),
):
    """Quick frame analysis summary without visualization with load balancing."""
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
            print(
                f"[{request_id}] Starting frame analysis with model: {target_model_name}"
            )

            # Check if model supports frame analysis with graceful error
            if not hasattr(model, "get_frame_analysis_summary"):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "error": f"Model '{target_model_name}' does not support frame analysis.",
                        "supported_methods": [
                            m
                            for m in dir(model)
                            if not m.startswith("_") and callable(getattr(model, m))
                        ],
                        "suggestion": "Use SIGLIP-LSTM-V3 or ColorCues models for frame analysis",
                        "available_models": list(settings.models.keys()),
                    },
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
                    lambda: safe_model_execution(
                        model, "get_frame_analysis_summary", temp_path
                    ),
                )

                if "error" in execution_result:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail={
                            "error": execution_result["error"],
                            "model_used": target_model_name,
                            "supported_methods": execution_result.get(
                                "supported_methods", []
                            ),
                        },
                    )

                summary_dict = execution_result["result"]

                if "error" in summary_dict:
                    raise ValueError(summary_dict["error"])

                return FrameAnalysisResponse(
                    success=True,
                    video_id=video_id,
                    summary=FrameAnalysisSummary(**summary_dict),
                )
            except HTTPException:
                raise
            except Exception as e:
                print(f"[{request_id}] Frame analysis failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": f"Frame analysis failed: {str(e)}",
                        "model_used": target_model_name,
                        "request_id": request_id,
                    },
                )
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

        finally:
            # Decrement request count
            with request_lock:
                request_counts[target_model_name] = max(
                    0, request_counts[target_model_name] - 1
                )


# Helper functions
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
        default="",
        description="Optional model name (defaults to load-balanced selection).",
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
            print(
                f"[{request_id}] Starting visualization with model: {target_model_name}"
            )

            # Check if model supports visualization
            if not hasattr(model, "predict_visualized"):
                return {
                    "success": False,
                    "error": f"Model '{target_model_name}' does not support visualization.",
                    "supported_methods": [
                        m
                        for m in dir(model)
                        if not m.startswith("_") and callable(getattr(model, m))
                    ],
                    "suggestion": "Try using SIGLIP-LSTM-V3 or ColorCues models for visualization",
                    "available_models": list(settings.models.keys()),
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
                    lambda: safe_model_execution(
                        model, "predict_visualized", temp_path
                    ),
                )

                if "error" in execution_result:
                    return {
                        "success": False,
                        "error": execution_result["error"],
                        "model_used": target_model_name,
                        "supported_methods": execution_result.get(
                            "supported_methods", []
                        ),
                    }

                visualization_path = execution_result["result"]

                return StreamingResponse(
                    file_iterator(visualization_path),
                    media_type="video/mp4",
                    headers={
                        "Content-Disposition": "attachment; filename=analysis_visualization.mp4"
                    },
                    background=BackgroundTask(
                        cleanup_files, temp_path, visualization_path
                    ),
                )

            except Exception as e:
                print(f"[{request_id}] Visualization failed: {e}")
                return {
                    "success": False,
                    "error": f"Visualization generation failed: {str(e)}",
                    "model_used": target_model_name,
                    "request_id": request_id,
                }
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

        finally:
            # Decrement request count
            with request_lock:
                request_counts[target_model_name] = max(
                    0, request_counts[target_model_name] - 1
                )


# Load Balancing Status Endpoint
@app.get("/status/load-balancing", tags=["Status"])
async def get_load_balancing_status():
    """Get current load balancing status and request counts."""
    with request_lock:
        return {
            "request_counts": dict(request_counts),
            "available_models": list(settings.models.keys()),
            "default_model": settings.default_model_name,
            "semaphore_limits": {
                model: sem._value for model, sem in request_semaphores.items()
            },
        }
