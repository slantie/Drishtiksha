# src/app/dependencies.py

import os
import tempfile
import logging
import aiofiles
from typing import Tuple, Dict, Any, AsyncGenerator
from fastapi import Depends, HTTPException, status, UploadFile, Form

from src.config import settings
from src.ml.registry import ModelManager

logger = logging.getLogger(__name__)

app_state: Dict[str, Any] = {}

def get_model_manager() -> ModelManager:
    """Dependency to get the singleton ModelManager instance."""
    manager = app_state.get("model_manager")
    if not manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ModelManager is not available. The service may be starting up or in a failed state."
        )
    return manager


# REFACTOR: Renamed from process_video_request to be media-agnostic.
async def process_media_request(
    # REFACTOR: Changed alias from 'video' to 'media' for clarity.
    media: UploadFile = Form(..., alias="media"),
    model_name_form: str = Form(
        default=None,
        alias="model",
        description="Optional: Specify an active model. If omitted, the default model is used."
    ),
    model_manager: ModelManager = Depends(get_model_manager)
) -> AsyncGenerator[Tuple[str, str], None]:
    """
    A robust dependency that handles the lifecycle of a media analysis request:
    1. Validates that the requested model is active and available.
    2. Securely saves the uploaded file to a temporary location in memory-efficient chunks.
    3. Yields the model name and the temporary file path to the endpoint.
    4. Guarantees cleanup of the temporary file after the request is complete.
    """
    target_model_name = model_name_form or settings.default_model_name

    if target_model_name not in model_manager.get_available_models():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Model '{target_model_name}' is not available. "
                f"Please choose from: {model_manager.get_available_models()}"
            )
        )

    temp_path = None
    try:
        suffix = os.path.splitext(media.filename)[1] if media.filename else ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name

        async with aiofiles.open(temp_path, 'wb') as out_file:
            content_written = 0
            while content := await media.read(1024 * 1024):  # Read in 1MB chunks
                await out_file.write(content)
                content_written += len(content)

        if content_written == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded media file is empty."
            )

        yield target_model_name, temp_path

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as e:
                logger.error(f"Error cleaning up temporary file {temp_path}: {e}")