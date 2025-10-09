# src/app/routers/analysis.py

import os
import shutil
import asyncio
import logging
from uuid import uuid4
from urllib.parse import urljoin
from fastapi import APIRouter, Depends, HTTPException, status, Form, Request

from src.app.dependencies import get_model_manager, process_media_request
from src.app.schemas import APIResponse, AnalysisResult, VideoAnalysisResult, AudioAnalysisResult
from src.app.security import get_api_key
from src.ml.registry import ModelManager
from src.ml.base import AnalysisResult as BaseModelAnalysisResult
from src.ml.exceptions import MediaProcessingError, InferenceError
from src.ml.correlation import get_correlation_id
from src.config import settings # Import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/analyze",
    tags=["Analysis"],
    dependencies=[Depends(get_api_key)]
)

MediaProcessingDeps = Depends(process_media_request)

def _log_analysis_result(model_name: str, media_path: str, result: BaseModelAnalysisResult):
    """A standardized helper function to log successful analysis results."""
    correlation_id = get_correlation_id() or "unknown"
    header = f"✅ {result.media_type.upper()} ANALYSIS COMPLETE"
    log_message = [
        f"| Correlation ID: {correlation_id}",
        f"| Media: {os.path.basename(media_path)}",
        f"| Model: {model_name}",
        f"| Prediction: {result.prediction}",
        f"| Confidence: {result.confidence:.4f}",
        f"| Processing Time: {result.processing_time:.2f}s",
    ]
    if result.note:
        log_message.append(f"| Note: {result.note}")
    logger.info("\n" + "\n".join(log_message))


async def _try_fallback_models(
    model_manager: ModelManager,
    primary_model_name: str,
    media_path: str,
    media_id: str,
    user_id: str,
    media_type: str,
    generate_visualizations: bool = False
) -> tuple[BaseModelAnalysisResult, str]:
    """
    Attempt to use fallback models if the primary model fails.
    
    Args:
        model_manager: ModelManager instance
        primary_model_name: The name of the model that failed
        media_path: Path to media file
        media_id: Media ID for event publishing
        user_id: User ID for event publishing
        media_type: Type of media ('video' or 'audio')
        generate_visualizations: Whether to generate visualization videos
        
    Returns:
        Tuple of (result, model_name_used) if successful
        
    Raises:
        Exception if all fallbacks fail
    """
    correlation_id = get_correlation_id() or "unknown"
    
    # Get all available models of the same type
    available_models = model_manager.get_available_models()
    configs = model_manager.get_active_model_configs()
    
    # Filter models by media type
    fallback_candidates = []
    for model_name in available_models:
        if model_name == primary_model_name:
            continue  # Skip the failed model
            
        config = configs.get(model_name)
        if config is None:
            continue
            
        # Check if model matches required media type
        if media_type == 'video' and getattr(config, 'isVideo', False):
            fallback_candidates.append(model_name)
        elif media_type == 'audio' and getattr(config, 'isAudio', False):
            fallback_candidates.append(model_name)
            
    if not fallback_candidates:
        logger.warning(
            f"[{correlation_id}] No fallback models available for {media_type} analysis"
        )
        raise Exception("No fallback models available")
        
    logger.info(
        f"[{correlation_id}] Attempting fallback models: {fallback_candidates}"
    )
    
    # Try each fallback model
    last_error = None
    for fallback_name in fallback_candidates:
        try:
            logger.info(f"[{correlation_id}] Trying fallback model: {fallback_name}")
            
            fallback_model = model_manager.get_model(fallback_name)
            if fallback_model is None:
                logger.warning(f"[{correlation_id}] Fallback model {fallback_name} not loaded")
                continue
                
            result = await asyncio.to_thread(
                fallback_model.analyze,
                media_path,
                generate_visualizations=generate_visualizations,
                video_id=media_id,
                user_id=user_id
            )
            
            # Add note about fallback usage
            if hasattr(result, 'note'):
                fallback_note = f"Primary model '{primary_model_name}' failed. Used fallback model '{fallback_name}'."
                result.note = f"{result.note} | {fallback_note}" if result.note else fallback_note
                
            logger.info(
                f"[{correlation_id}] Fallback successful with model: {fallback_name}"
            )
            
            return result, fallback_name
            
        except Exception as e:
            logger.warning(
                f"[{correlation_id}] Fallback model {fallback_name} also failed: {e}"
            )
            last_error = e
            continue
            
    # All fallbacks failed
    raise Exception(f"All fallback models failed. Last error: {last_error}")

@router.post(
    "",
    response_model=APIResponse[AnalysisResult],
    summary="Perform Comprehensive Media Analysis"
)
async def analyze_media(
    request: Request,
    model_manager: ModelManager = Depends(get_model_manager),
    proc_data: tuple = MediaProcessingDeps,
    media_id: str = Form(None, alias="mediaId"),
    user_id: str = Form(None, alias="userId"),
):
    """The new, single entry point for all analysis with robust error handling and graceful degradation."""
    correlation_id = get_correlation_id() or "unknown"
    model_name, temp_media_path = proc_data
    model = model_manager.get_model(model_name)
    
    logger.info(
        f"[{correlation_id}] Starting analysis with model: {model_name}, "
        f"media: {os.path.basename(temp_media_path)}"
    )

    try:
        # For API calls, we don't generate visualization videos (only raw data)
        # CLI can explicitly set generate_visualizations=True
        result = await asyncio.to_thread(
            model.analyze, 
            temp_media_path, 
            generate_visualizations=False,  # API default: no video generation
            video_id=media_id, 
            user_id=user_id
        )
        
        # MODIFIED: Handle moving visualization files to permanent storage
        temp_vis_path = None
        if isinstance(result, VideoAnalysisResult):
            temp_vis_path = result.visualization_path
        elif isinstance(result, AudioAnalysisResult) and result.visualization:
            temp_vis_path = result.visualization.spectrogram_url

        if temp_vis_path and os.path.exists(temp_vis_path):
            vis_subfolder = "visualizations"
            permanent_vis_dir = settings.storage_path / vis_subfolder
            permanent_vis_dir.mkdir(exist_ok=True)
            
            unique_filename = f"{uuid4()}{os.path.splitext(temp_vis_path)[1]}"
            permanent_vis_path = permanent_vis_dir / unique_filename
            
            shutil.move(temp_vis_path, permanent_vis_path)

            # Construct the final public URL
            url_path_segment = settings.storage_path.name
            download_url = urljoin(settings.assets_base_url, f"{url_path_segment}/{vis_subfolder}/{unique_filename}")
            
            if isinstance(result, VideoAnalysisResult):
                result.visualization_path = download_url
            elif isinstance(result, AudioAnalysisResult):
                result.visualization.spectrogram_url = download_url
            
            logger.info(f"Moved visualization to {permanent_vis_path} and generated URL: {download_url}")

        _log_analysis_result(model_name, temp_media_path, result)
        return APIResponse(model_used=model_name, data=result)

    except MediaProcessingError as e:
        logger.warning(
            f"[{correlation_id}] Analysis failed: Invalid media file. Error: {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "Media Processing Error",
                "message": f"Media could not be processed. Reason: {e}",
                "correlation_id": correlation_id
            }
        )
    except InferenceError as e:
        logger.error(
            f"[{correlation_id}] Inference error with model {model_name}: {e}",
            exc_info=True
        )
        
        # Attempt graceful degradation with fallback models
        try:
            logger.info(f"[{correlation_id}] Attempting graceful degradation...")
            
            # Determine media type from result or config
            config = model_manager.get_active_model_configs().get(model_name)
            media_type = 'video' if getattr(config, 'isVideo', False) else 'audio'
            
            result, fallback_model_name = await _try_fallback_models(
                model_manager,
                model_name,
                temp_media_path,
                media_id,
                user_id,
                media_type,
                generate_visualizations=False  # Fallback also skips visualization
            )
            
            # Handle visualization moving (same logic as before)
            temp_vis_path = None
            if isinstance(result, VideoAnalysisResult):
                temp_vis_path = result.visualization_path
            elif isinstance(result, AudioAnalysisResult) and result.visualization:
                temp_vis_path = result.visualization.spectrogram_url

            if temp_vis_path and os.path.exists(temp_vis_path):
                vis_subfolder = "visualizations"
                permanent_vis_dir = settings.storage_path / vis_subfolder
                permanent_vis_dir.mkdir(exist_ok=True)
                
                unique_filename = f"{uuid4()}{os.path.splitext(temp_vis_path)[1]}"
                permanent_vis_path = permanent_vis_dir / unique_filename
                
                shutil.move(temp_vis_path, permanent_vis_path)

                url_path_segment = settings.storage_path.name
                download_url = urljoin(settings.assets_base_url, f"{url_path_segment}/{vis_subfolder}/{unique_filename}")
                
                if isinstance(result, VideoAnalysisResult):
                    result.visualization_path = download_url
                elif isinstance(result, AudioAnalysisResult):
                    result.visualization.spectrogram_url = download_url
            
            _log_analysis_result(fallback_model_name, temp_media_path, result)
            return APIResponse(model_used=fallback_model_name, data=result)
            
        except Exception as fallback_error:
            logger.error(
                f"[{correlation_id}] All fallback attempts failed: {fallback_error}",
                exc_info=True
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Inference Error",
                    "message": f"Model inference failed and all fallbacks exhausted.",
                    "primary_error": str(e),
                    "fallback_error": str(fallback_error),
                    "correlation_id": correlation_id
                }
            )
            
    except Exception as e:
        logger.error(
            f"[{correlation_id}] Unknown critical error during analysis: {e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal Server Error",
                "message": "An unknown server error occurred.",
                "correlation_id": correlation_id
            }
        )

# REMOVED: This endpoint is no longer needed as the asset server handles all file serving.
# @router.get("/visualization/{filename}", ...)