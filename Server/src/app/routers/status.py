# src/app/routers/status.py

from fastapi import APIRouter, Depends

from src.app.dependencies import get_model_manager
from src.app.schemas import HealthStatus, ModelStatus, ServerStats, DeviceInfo, SystemInfo
from src.app.security import get_api_key
from src.ml.registry import ModelManager
from src.ml.system_info import system_monitor
from src.ml.health_check import HealthChecker
from src.ml.correlation import set_correlation_id, get_correlation_id
from src.config import settings

# --- Router for Public Endpoints ---
# This router has NO security dependency and is for endpoints that need to be open,
# like health checks for load balancers or container orchestrators.
public_router = APIRouter(tags=["Status & Statistics"])

# --- Router for Private, Protected Endpoints ---
# This router applies the API key security dependency to ALL endpoints defined on it.
private_router = APIRouter(
    tags=["Status & Statistics"],
    dependencies=[Depends(get_api_key)]
)

# --- Public Endpoints ---

@public_router.get("/", response_model=HealthStatus)
def get_root_health(manager: ModelManager = Depends(get_model_manager)):
    """Provides a public health check of the service and the status of all active models."""
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

@public_router.get("/ping")
def ping():
    """A simple public ping endpoint to confirm the server is running."""
    return {"status": "pong"}

# --- Private, API Key-Protected Endpoints ---

@private_router.get("/stats", response_model=ServerStats)
def get_server_stats(manager: ModelManager = Depends(get_model_manager)):
    """(Protected) Get comprehensive server statistics including device, system, and model information."""
    system_info_data = system_monitor.get_system_info()
    
    return ServerStats(
        service_name=settings.project_name,
        version="3.0.0",
        status="running",
        uptime_seconds=system_info_data.uptime_seconds,
        device_info=system_monitor.get_device_info(),
        system_info=system_info_data,
        models_info=system_monitor.get_models_info(manager),
        active_models_count=len(manager.get_loaded_model_names()),
        total_models_count=len(manager.get_active_model_configs()),
        configuration=system_monitor.get_server_configuration()
    )

@private_router.get("/device", response_model=DeviceInfo)
def get_device_status():
    """(Protected) Get detailed information about the compute device (GPU/CPU)."""
    return system_monitor.get_device_info()

@private_router.get("/system", response_model=SystemInfo)
def get_system_status():
    """(Protected) Get system resource information (RAM, CPU, etc.)."""
    return system_monitor.get_system_info()

@private_router.get("/models")
def get_models_info(manager: ModelManager = Depends(get_model_manager)):
    """(Protected) Get detailed information about all configured and loaded models."""
    return {
        "models": system_monitor.get_models_info(manager),
        "summary": {
            "total_configured": len(manager.get_active_model_configs()),
            "currently_loaded": len(manager.get_loaded_model_names()),
            "active_models": manager.get_available_models(),
            "loaded_models": manager.get_loaded_model_names()
        }
    }

@private_router.get("/config")
def get_configuration():
    """(Protected) Get server configuration summary."""
    return system_monitor.get_server_configuration()

@private_router.get("/health/deep")
async def get_deep_health_check(
    force_refresh: bool = False,
    manager: ModelManager = Depends(get_model_manager)
):
    """
    (Protected) Perform comprehensive health check on all models.
    
    This endpoint validates that each model can:
    - Load successfully
    - Perform inference
    - Return proper results
    
    Args:
        force_refresh: If True, bypass cache and perform fresh checks
        
    Returns:
        Detailed health status for all models
    """
    # Set correlation ID for request tracing
    import uuid
    correlation_id = str(uuid.uuid4())
    set_correlation_id(correlation_id)
    
    health_checker = HealthChecker(manager)
    results = await health_checker.check_all_models(force_refresh=force_refresh)
    results["correlation_id"] = correlation_id
    
    return results

@private_router.post("/health/clear-cache")
def clear_health_cache(manager: ModelManager = Depends(get_model_manager)):
    """(Protected) Clear the health check cache."""
    health_checker = HealthChecker(manager)
    health_checker.clear_cache()
    return {"status": "success", "message": "Health check cache cleared"}