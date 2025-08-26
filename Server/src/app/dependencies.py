# src/app/dependencies.py

import os
import uuid
import tempfile
import logging
import aiofiles
from typing import Generator, Tuple, Dict, Any, AsyncGenerator
from fastapi import Depends, HTTPException, status, UploadFile, Form

from src.config import settings
from src.ml.registry import ModelManager

logger = logging.getLogger(__name__)

# --- State Management ---
# This dictionary will hold our application's state, like the ModelManager instance.
# It will be populated during the startup event in main.py.
app_state: Dict[str, Any] = {}

def get_model_manager() -> ModelManager:
    """Dependency to get the singleton ModelManager instance."""
    manager = app_state.get("model_manager")
    if not manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ModelManager is not available. The service may be starting up."
        )
    return manager

# --- "Super Dependency" for Analysis Endpoints ---
# This dependency handles all the boilerplate logic for analysis requests.
# --- "Super Dependency" for Analysis Endpoints (UPDATED) ---
async def process_video_request(
    video: UploadFile,
    model_name_form: str = Form(
        default=None,
        alias="model",
        description="Optional: Specify a model. If omitted, the default model is chosen."
    ),
    model_manager: ModelManager = Depends(get_model_manager)
) -> AsyncGenerator[Tuple[str, str], None]:
    """
    A dependency that handles the entire lifecycle of a media analysis request:
    1. Selects the appropriate model.
    2. Securely and robustly saves the uploaded file to a temporary location.
    3. Yields the model name and the path to the temporary file to the endpoint.
    4. Cleans up the temporary file after the request is complete.
    """
    target_model_name = model_name_form or settings.default_model_name
    
    if target_model_name not in model_manager.get_available_models():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Model '{target_model_name}' not found."
        )

    temp_path = None
    try:
        # --- START OF REFACTORED FILE WRITING LOGIC ---
        # Create a secure temporary file with the correct suffix
        suffix = os.path.splitext(video.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name

        # Write the uploaded file content in chunks to the temporary file
        # This is more memory-efficient and robust than reading the whole file at once.
        async with aiofiles.open(temp_path, 'wb') as out_file:
            content_written = 0
            while content := await video.read(1024 * 1024):  # Read in 1MB chunks
                await out_file.write(content)
                content_written += len(content)

        # Check if any content was written to the file
        if content_written == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file cannot be empty.")
        
        # --- END OF REFACTORED FILE WRITING LOGIC ---

        # Yield control back to the endpoint function with the guaranteed-to-be-complete file
        yield target_model_name, temp_path

    finally:
        # This cleanup code runs after the response has been sent
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as e:
                logger.error(f"Error cleaning up temp file {temp_path}: {e}")