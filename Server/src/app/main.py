# src/app/main.py

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.config import settings
from src.ml.registry import ModelManager
from src.app.dependencies import app_state
from src.app.routers import analysis, status
from src.app.schemas import APIError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S | %d/%m/%Y"
)

# Reduce uvicorn logging verbosity
logging.getLogger("uvicorn").setLevel(logging.ERROR)
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
logging.getLogger("uvicorn.access").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan.
    - On startup: Initializes the ModelManager and pre-loads ALL configured models.
    - On shutdown: Cleans up resources.
    """
    logger.info(f"🔼 Starting up {settings.project_name} Application.")
    manager = ModelManager(settings)
    app_state["model_manager"] = manager
    
    # --- CHANGE: Loop through and pre-load ALL models ---
    all_models = manager.get_available_models()
    for model_name in all_models:
        try:
            manager.get_model(model_name)
        except Exception as e:
            logger.critical(
                f"❌ CRITICAL: Failed to load model '{model_name}' during startup: {e}", 
                exc_info=True
            )

    logger.info(f"✅ {settings.project_name} Application Ready.")
    yield
    
    logger.info("🔽 Shutting down server.")
    app_state.clear()

# --- FastAPI App Initialization ---
app = FastAPI(
    title=settings.project_name,
    version="2.0.0",
    lifespan=lifespan
)

# --- Custom Exception Handlers ---
@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=422, # Use 422 for Unprocessable Entity
        content=APIError(error="Validation Error", message=str(exc)).model_dump(),
    )

@app.exception_handler(NotImplementedError)
async def not_implemented_error_handler(request: Request, exc: NotImplementedError):
    return JSONResponse(
        status_code=501, # 501 Not Implemented
        content=APIError(error="Not Implemented", message=str(exc)).model_dump(),
    )

# --- Include Routers ---
app.include_router(status.router)
app.include_router(analysis.router)