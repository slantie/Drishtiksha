# src/app/routers/analysis.py

import os
import asyncio
import logging
import tempfile
from urllib.parse import urljoin
from fastapi import APIRouter, Depends, HTTPException, status, Form, Request
from fastapi.responses import StreamingResponse

from src.app.dependencies import get_model_manager, process_media_request
# FIX: The import for AnalysisResult will now work correctly.
from src.app.schemas import APIResponse, AnalysisResult, VideoAnalysisResult, AudioAnalysisResult
from src.app.security import get_api_key
from src.ml.registry import ModelManager
from src.ml.base import AnalysisResult as BaseModelAnalysisResult
from src.ml.exceptions import MediaProcessingError, InferenceError

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
    user_id: str = Form(None),
):
    """The new, single entry point for all analysis with robust error handling."""
    model_name, media_path = proc_data
    model = model_manager.get_model(model_name)

    try:
        result = await asyncio.to_thread(
            model.analyze, media_path, video_id=media_id, user_id=user_id
        )
        vis_path = None
        if isinstance(result, VideoAnalysisResult):
            vis_path = result.visualization_path
        elif isinstance(result, AudioAnalysisResult) and result.visualization:
            vis_path = result.visualization.spectrogram_url

        if vis_path:
            filename = os.path.basename(vis_path)
            base_url = str(request.base_url)
            download_url = urljoin(base_url, f"analyze/visualization/{filename}")
            
            if isinstance(result, VideoAnalysisResult):
                result.visualization_path = download_url
            elif isinstance(result, AudioAnalysisResult):
                result.visualization.spectrogram_url = download_url
            
            logger.info(f"Generated visualization, accessible at: {download_url}")

        _log_analysis_result(model_name, media_path, result)
        return APIResponse(model_used=model_name, data=result)

    except MediaProcessingError as e:
        logger.warning(f"Analysis failed for '{os.path.basename(media_path)}' with model '{model_name}': Invalid media file. Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"The uploaded media file could not be processed. It may be corrupt or in an unsupported format. Reason: {e}"
        )
    except InferenceError as e:
        logger.error(f"Critical inference error for '{os.path.basename(media_path)}' with model '{model_name}'. Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during model inference. Please contact support. Error ID: {media_id or 'N/A'}"
        )
    except Exception as e:
        logger.error(f"An unknown critical error occurred during analysis of '{os.path.basename(media_path)}' with model '{model_name}'. Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unknown server error occurred. Please try again later."
        )


@router.get(
    "/visualization/{filename}",
    summary="Download Generated Visualization"
)
async def download_visualization(filename: str):
    """Downloads a previously generated visualization file by its filename."""
    # FIX: Replace incorrect security import with standard, safe os.path.basename
    # This prevents path traversal attacks by stripping all directory information.
    clean_filename = os.path.basename(filename)

    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, clean_filename)

    if not os.path.exists(file_path):
        logger.warning(f"Visualization download request for non-existent file: {clean_filename}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Visualization file '{clean_filename}' not found or has expired."
        )

    logger.info(f"📥 Serving download for visualization file: {clean_filename}")

    async def file_iterator_with_cleanup(path: str):
        # The 'aiofiles' library is not a default dependency, so we use standard file I/O
        # within an async function for compatibility and simplicity.
        with open(path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                yield chunk
        try:
            os.remove(path)
            logger.debug(f"Cleaned up temporary visualization file after download: {path}")
        except OSError as e:
            logger.error(f"Error cleaning up temp file {path} after download: {e}")

    media_type = "video/mp4" if clean_filename.endswith(".mp4") else "image/png"
    return StreamingResponse(
        file_iterator_with_cleanup(file_path),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={clean_filename}"}
    )