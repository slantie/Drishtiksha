# src/app/dependencies.py

import os
import tempfile
import logging
import aiofiles
from typing import Tuple, Dict, Any, AsyncGenerator
from fastapi import Depends, HTTPException, status, UploadFile, Form

from src.config import settings
from src.ml.registry import ModelManager
from src.ml.utils import get_media_type

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

async def process_media_request(
    media: UploadFile = Form(..., alias="media"),
    model_name_form: str = Form(
        default=None,
        alias="model",
        description="Optional: Specify a model. If omitted, the server will select one based on media type."
    ),
    model_manager: ModelManager = Depends(get_model_manager)
) -> AsyncGenerator[Tuple[str, str], None]:
    """
    REFACTORED with enhanced logging for robust model selection.
    """
    temp_path = None
    try:
        # Step 1: Save file to determine its type
        suffix = os.path.splitext(media.filename)[1] if media.filename else ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name

        async with aiofiles.open(temp_path, 'wb') as out_file:
            content_written = 0
            while content := await media.read(1024 * 1024):
                await out_file.write(content)
                content_written += len(content)

        if content_written == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded media file is empty."
            )

        # Step 2: Determine media type
        media_type = get_media_type(temp_path)
        if media_type == "UNKNOWN":
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Could not determine media type for file '{media.filename}'."
            )

        # Step 3: Select and validate the model
        target_model_name = None
        active_model_configs = model_manager.get_active_model_configs()
        
        if model_name_form:
            if model_name_form not in active_model_configs:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Requested model '{model_name_form}' is not active or available."
                )
            target_model_name = model_name_form
        else:
            # --- REFACTORED AUTO-SELECTION LOGIC ---
            logger.info(f"Auto-selecting model for media type: {media_type}")
            
            suitable_models = [
                name for name, config in active_model_configs.items()
                if (media_type == "IMAGE" and config.isImage) or \
                   (media_type == "VIDEO" and config.isVideo) or \
                   (media_type == "AUDIO" and config.isAudio)
            ]
            
            logger.info(f"Found suitable models: {suitable_models}")
            logger.info(f"Comparing against default model from settings: '{settings.default_model_name}'")

            if not suitable_models:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"No active models were found that can process media of type '{media_type}'."
                )
            
            # Explicitly check if the configured default model is in the list of suitable models.
            if settings.default_model_name in suitable_models:
                logger.info(f"Default model '{settings.default_model_name}' is suitable. Selecting it.")
                target_model_name = settings.default_model_name
            else:
                target_model_name = suitable_models[0]
                logger.warning(
                    f"Default model '{settings.default_model_name}' is NOT suitable for '{media_type}'. "
                    f"Falling back to first available model: '{target_model_name}'."
                )

        # Final validation remains the same...
        chosen_model_config = active_model_configs.get(target_model_name)
        if not chosen_model_config:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not select a valid model for the request."
            )

        is_supported = (
            (media_type == "IMAGE" and chosen_model_config.isImage) or
            (media_type == "VIDEO" and chosen_model_config.isVideo) or
            (media_type == "AUDIO" and chosen_model_config.isAudio)
        )
        if not is_supported:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"The selected model '{target_model_name}' does not support media of type '{media_type}'."
            )

        logger.info(f"Final selected model: '{target_model_name}' for media type '{media_type}'.")
        yield target_model_name, temp_path

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as e:
                logger.error(f"Error cleaning up temporary file {temp_path}: {e}")