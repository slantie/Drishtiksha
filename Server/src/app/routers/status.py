# src/app/routers/status.py

from fastapi import APIRouter, Depends

from src.app.dependencies import get_model_manager
from src.app.schemas import HealthStatus, ModelStatus, ServerStats, DeviceInfo, SystemInfo
from src.ml.registry import ModelManager
from src.ml.system_info import get_device_info, get_system_info, get_model_info, get_server_configuration
from src.config import settings

router = APIRouter(tags=["Status & Statistics"])

@router.get("/", response_model=HealthStatus)
def get_root_health(manager: ModelManager = Depends(get_model_manager)):
    """Provides a detailed health check of the service and the status of all active models."""
    # REFACTOR: Use public methods to get model info, avoiding private member access.
    loaded_model_names = manager.get_loaded_model_names()
    active_model_configs = manager.get_active_model_configs()

    model_statuses = [
        ModelStatus(
            name=name,
            loaded=(name in loaded_model_names),
            description=config.description
        ) for name, config in active_model_configs.items()
    ]
        
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
    # REFACTOR: Uptime is now fetched from the centralized system_info module.
    system_info_data = get_system_info()
    
    return ServerStats(
        service_name=settings.project_name,
        version="3.0.0",
        status="running",
        uptime_seconds=system_info_data.uptime_seconds,
        device_info=get_device_info(),
        system_info=system_info_data,
        models_info=get_model_info(manager),
        active_models_count=len(manager.get_loaded_model_names()),
        total_models_count=len(manager.get_active_model_configs()),
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
            "total_configured": len(manager.get_active_model_configs()),
            "currently_loaded": len(manager.get_loaded_model_names()),
            "active_models": manager.get_available_models(),
            "loaded_models": manager.get_loaded_model_names()
        }
    }

@router.get("/config")
def get_configuration():
    """Get server configuration summary."""
    return get_server_configuration()