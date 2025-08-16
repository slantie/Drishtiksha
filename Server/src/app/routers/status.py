# src/app/routers/status.py

from fastapi import APIRouter, Depends

from src.app.dependencies import get_model_manager
from src.app.schemas import HealthStatus, ModelStatus, ServerStats, DeviceInfo, SystemInfo
from src.ml.registry import ModelManager
from src.ml.system_info import get_device_info, get_system_info, get_model_info, get_server_configuration
from src.config import settings
import time

router = APIRouter(tags=["Status & Statistics"])

# Track server start time
_server_start_time = time.time()

@router.get("/", response_model=HealthStatus)
def get_root_health(manager: ModelManager = Depends(get_model_manager)):
    """Provides a detailed health check of the service and loaded models."""
    loaded_models = manager._models # Accessing private member for status, OK here
    all_model_configs = manager.model_configs

    model_statuses = []
    for name, config in all_model_configs.items():
        model_statuses.append(ModelStatus(
            name=name,
            loaded=(name in loaded_models),
            description=config.description
        ))
        
    return HealthStatus(
        status="ok",
        active_models=model_statuses,
        default_model=settings.default_model_name,
    )

@router.get("/ping")
def ping():
    """A simple ping endpoint to confirm the server is running."""
    return {"status": "pong"}

@router.get("/stats", response_model=ServerStats)
def get_server_stats(manager: ModelManager = Depends(get_model_manager)):
    """Get comprehensive server statistics including device, system, and model information."""
    uptime_seconds = time.time() - _server_start_time
    
    return ServerStats(
        service_name=settings.project_name,
        version="2.0.0",
        status="running",
        uptime_seconds=round(uptime_seconds, 2),
        device_info=get_device_info(),
        system_info=get_system_info(),
        models_info=get_model_info(manager),
        active_models_count=len([m for m in manager._models.keys()]),
        total_models_count=len(manager.model_configs),
        configuration=get_server_configuration()
    )

@router.get("/device", response_model=DeviceInfo)
def get_device_status():
    """Get detailed information about the compute device (GPU/CPU)."""
    return get_device_info()

@router.get("/system", response_model=SystemInfo)
def get_system_status():
    """Get system resource information (RAM, CPU, etc.)."""
    return get_system_info()

@router.get("/models")
def get_models_info(manager: ModelManager = Depends(get_model_manager)):
    """Get detailed information about all configured and loaded models."""
    return {
        "models": get_model_info(manager),
        "summary": {
            "total_configured": len(manager.model_configs),
            "currently_loaded": len(manager._models),
            "active_models": list(manager.model_configs.keys()),
            "loaded_models": list(manager._models.keys())
        }
    }

@router.get("/config")
def get_configuration():
    """Get server configuration summary."""
    return get_server_configuration()