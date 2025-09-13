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
    header = f"✅ {result.media_type.upper()} ANALYSIS COMPLETE"
    log_message = [
        f"| Media: {os.path.basename(media_path)}",
        f"| Model: {model_name}",
        f"| Prediction: {result.prediction}",
        f"| Confidence: {result.confidence:.4f}",
        f"| Processing Time: {result.processing_time:.2f}s",
    ]
    if result.note:
        log_message.append(f"| Note: {result.note}")
    logger.info("\n" + "\n".join(log_message))

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
    """The new, single entry point for all analysis with robust error handling."""
    model_name, temp_media_path = proc_data
    model = model_manager.get_model(model_name)

    try:
        result = await asyncio.to_thread(
            model.analyze, temp_media_path, video_id=media_id, user_id=user_id
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
        logger.warning(f"Analysis failed: Invalid media file. Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Media could not be processed. Reason: {e}"
        )
    except InferenceError as e:
        logger.error(f"Critical inference error. Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during model inference."
        )
    except Exception as e:
        logger.error(f"An unknown critical error occurred during analysis. Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unknown server error occurred."
        )

# REMOVED: This endpoint is no longer needed as the asset server handles all file serving.
# @router.get("/visualization/{filename}", ...)