# src/app/routers/status.py

from fastapi import APIRouter, Depends

from src.app.dependencies import get_model_manager
from src.app.schemas import HealthStatus, ModelStatus
from src.ml.registry import ModelManager
from src.config import settings

router = APIRouter(tags=["Status"])

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