# src/app/main.py

import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.config import settings
from src.ml.registry import ModelManager, ModelRegistryError
from src.app.dependencies import app_state
from src.app.routers import analysis, status
from src.app.schemas import APIError

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if root_logger.hasHandlers():
    root_logger.handlers.clear()
log_format = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
date_format = "%H:%M:%S"
formatter = logging.Formatter(log_format, datefmt=date_format)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("h5py").setLevel(logging.WARNING)
# Set TensorFlow logging to ERROR to reduce verbosity
logging.getLogger("tensorflow").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan, ensuring robust startup and shutdown.
    """
    logger.info(f"🔼 Starting up {settings.project_name}...")
    
    # 1. Initialize the Model Manager
    manager = ModelManager(settings)
    app_state["model_manager"] = manager

    # 2. Eagerly load all active models.
    try:
        # FIX: The incorrect for-loop has been removed.
        # This is now called only ONCE to load all models defined in the config.
        manager.load_models()
    except ModelRegistryError as e:
        # The manager already logs the detailed error. This is a final, fatal message.
        logger.critical(f"❌ FATAL: A critical model failed to load. The application cannot start. Reason: {e}")
        # Exit with a non-zero code to signal failure to container orchestrators.
        sys.exit(1)

    logger.info(f"✅ {settings.project_name} startup complete. All active models loaded.")
    yield
    
    logger.info(f"🔽 Shutting down {settings.project_name}.")
    app_state.clear()


# --- FastAPI App Initialization ---
app = FastAPI(
    title=settings.project_name,
    version="3.0.0",
    lifespan=lifespan,
    description="A high-performance inference server for deepfake detection models.",
    contact={
        "name": "Zaptrix",
        "email": "zaptrix.io@gmail.com",
    },
)

# --- Custom Exception Handlers ---

# ADD: A global "catch-all" exception handler for ultimate robustness.
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Handles any unexpected, unhandled exceptions that occur during a request.
    Ensures the client always receives a structured JSON error response.
    """
    # Log the full traceback for the unhandled exception for debugging.
    logger.critical(f"Unhandled exception during request to {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=APIError(
            error="Internal Server Error",
            message="An unexpected error occurred. The technical team has been notified."
        ).model_dump(),
    )

@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    logger.warning(f"Validation error for request {request.url}: {exc}", exc_info=False)
    return JSONResponse(
        status_code=422,
        content=APIError(error="Validation Error", message=str(exc)).model_dump(),
    )

# Note: The NotImplementedError handler is less critical now but is kept for good practice.
@app.exception_handler(NotImplementedError)
async def not_implemented_error_handler(request: Request, exc: NotImplementedError):
    logger.warning(f"Feature not implemented for request {request.url}: {exc}", exc_info=False)
    return JSONResponse(
        status_code=501,
        content=APIError(error="Not Implemented", message=str(exc)).model_dump(),
    )


# --- Include Routers ---
app.include_router(analysis.router)
app.include_router(status.public_router)
app.include_router(status.private_router)