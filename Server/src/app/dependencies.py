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
    Processes the uploaded media file, saves it temporarily, and intelligently
    selects the appropriate model for analysis, prioritizing specialists.
    """
    temp_path = None
    try:
        # Step 1: Save file to a temporary location
        suffix = os.path.splitext(media.filename)[1] if media.filename else ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name

        async with aiofiles.open(temp_path, 'wb') as out_file:
            content_written = 0
            while content := await media.read(1024 * 1024): # Read in 1MB chunks
                await out_file.write(content)
                content_written += len(content)

        if content_written == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded media file is empty."
            )

        # Step 2: Determine media type from the saved file
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
            # --- REVISED AUTO-SELECTION LOGIC ---
            logger.info(f"Auto-selecting model for media type: {media_type}")

            # 1. Prioritize Specialist Models (e.g., isImage=True, isVideo=False for images)
            specialist_models = [
                name for name, config in active_model_configs.items()
                if (media_type == "IMAGE" and config.isImage and not config.isVideo and not config.isAudio) or \
                   (media_type == "VIDEO" and config.isVideo and not config.isImage and not config.isAudio) or \
                   (media_type == "AUDIO" and config.isAudio and not config.isVideo and not config.isImage)
            ]

            if specialist_models:
                logger.info(f"Found specialist models: {specialist_models}")
                if settings.default_model_name in specialist_models:
                    target_model_name = settings.default_model_name
                else:
                    target_model_name = specialist_models[0]
            else:
                # 2. Fallback to Generalist Models
                logger.warning(f"No specialist model found for {media_type}. Searching for generalist models.")
                generalist_models = [
                    name for name, config in active_model_configs.items()
                    if (media_type == "IMAGE" and config.isImage) or \
                       (media_type == "VIDEO" and config.isVideo) or \
                       (media_type == "AUDIO" and config.isAudio)
                ]

                if not generalist_models:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"No active models found that can process media of type '{media_type}'."
                    )
                
                logger.info(f"Found generalist models: {generalist_models}")
                if settings.default_model_name in generalist_models:
                    target_model_name = settings.default_model_name
                else:
                    target_model_name = generalist_models[0]

        # Final validation
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