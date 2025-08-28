# src/app/main.py

import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.config import settings
from src.ml.registry import ModelManager
from src.app.dependencies import app_state
from src.app.routers import analysis, status
from src.app.schemas import APIError

# Configure logging for the main application
logging.basicConfig(
    level=logging.INFO, # General logging level
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S | %d/%m/%Y"
)

# Reduce uvicorn logging verbosity for a cleaner console
# Set to WARNING to only show significant events from Uvicorn itself
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan.
    - On startup: Initializes the ModelManager and pre-loads all ACTIVE models.
    - On shutdown: Cleans up resources.
    """
    logger.info(f"🔼 Starting up {settings.project_name}.")
    manager = ModelManager(settings)
    app_state["model_manager"] = manager
    
    # Loop through and pre-load only the ACTIVE models.
    for model_name in manager.get_available_models():
        try:
            # This call will load the model into memory if it hasn't been already.
            manager.get_model(model_name)
        except Exception as e:
            logger.critical(
                f"❌ FATAL: Failed to load critical model '{model_name}' during startup: {e}", 
                exc_info=True
            )
            # Explicitly exit if a model fails to load. This is crucial for container health checks.
            sys.exit(1)

    logger.info(f"✅ {settings.project_name} startup complete. All active models loaded.")
    yield
    
    logger.info("🔽 Shutting down server.")
    app_state.clear()

# --- FastAPI App Initialization ---
app = FastAPI(
    title=settings.project_name,
    version="3.0.0",
    lifespan=lifespan
)

# --- Custom Exception Handlers ---
@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    logger.warning(f"Validation error: {exc}", exc_info=False) # Log as warning, not error
    return JSONResponse(
        status_code=422, # Use 422 for Unprocessable Entity
        content=APIError(error="Validation Error", message=str(exc)).model_dump(),
    )

@app.exception_handler(NotImplementedError)
async def not_implemented_error_handler(request: Request, exc: NotImplementedError):
    logger.warning(f"Feature not implemented: {exc}", exc_info=False)
    return JSONResponse(
        status_code=501, # 501 Not Implemented
        content=APIError(error="Not Implemented", message=str(exc)).model_dump(),
    )

# --- Include Routers ---
app.include_router(status.router)
app.include_router(analysis.router)