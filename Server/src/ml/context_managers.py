# src/ml/context_managers.py

"""
Context managers for safe resource management in ML models.
These ensure proper cleanup even when exceptions occur.
"""

import cv2
import tempfile
import logging
from pathlib import Path
from typing import Optional, Generator
from contextlib import contextmanager

from src.ml.exceptions import VideoCapturError, TemporaryFileError

logger = logging.getLogger(__name__)


# =============================================================================
# VIDEO PROCESSING CONTEXT MANAGERS
# =============================================================================

@contextmanager
def video_capture(video_path: str) -> Generator[cv2.VideoCapture, None, None]:
    """
    Context manager for cv2.VideoCapture that ensures proper cleanup.
    
    Usage:
        with video_capture("video.mp4") as cap:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # ... use cap ...
        # cap.release() is automatically called
    
    Args:
        video_path: Path to the video file
        
    Yields:
        cv2.VideoCapture object
        
    Raises:
        VideoCapturError: If video cannot be opened
    """
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoCapturError(
                f"Could not open video file: {video_path}",
                details={"video_path": video_path}
            )
        logger.debug(f"Opened video capture for: {video_path}")
        yield cap
    except Exception as e:
        logger.error(f"Error in video_capture context: {e}")
        raise
    finally:
        if cap is not None:
            cap.release()
            logger.debug(f"Released video capture for: {video_path}")


@contextmanager
def video_writer(
    output_path: str,
    fourcc: str,
    fps: float,
    frame_size: tuple[int, int]
) -> Generator[cv2.VideoWriter, None, None]:
    """
    Context manager for cv2.VideoWriter that ensures proper cleanup.
    
    Usage:
        with video_writer("output.mp4", "mp4v", 30, (640, 480)) as writer:
            writer.write(frame)
            # ... write more frames ...
        # writer.release() is automatically called
    
    Args:
        output_path: Path for output video
        fourcc: FourCC code for video codec
        fps: Frames per second
        frame_size: (width, height) tuple
        
    Yields:
        cv2.VideoWriter object
    """
    writer = None
    try:
        fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
        writer = cv2.VideoWriter(output_path, fourcc_code, fps, frame_size)
        if not writer.isOpened():
            raise VideoCapturError(
                f"Could not create video writer for: {output_path}",
                details={"output_path": output_path, "fps": fps, "frame_size": frame_size}
            )
        logger.debug(f"Created video writer for: {output_path}")
        yield writer
    except Exception as e:
        logger.error(f"Error in video_writer context: {e}")
        raise
    finally:
        if writer is not None:
            writer.release()
            logger.debug(f"Released video writer for: {output_path}")


# =============================================================================
# TEMPORARY FILE CONTEXT MANAGERS
# =============================================================================

@contextmanager
def temporary_file(
    suffix: str = "",
    prefix: str = "temp_",
    dir: Optional[str] = None,
    delete: bool = True
) -> Generator[Path, None, None]:
    """
    Context manager for creating and managing temporary files.
    
    Usage:
        with temporary_file(suffix=".mp4") as temp_path:
            # Use temp_path for file operations
            with open(temp_path, 'wb') as f:
                f.write(data)
        # File is automatically deleted (if delete=True)
    
    Args:
        suffix: File extension (e.g., ".mp4", ".wav")
        prefix: Filename prefix
        dir: Directory for temp file (None = system temp)
        delete: Whether to delete file after use
        
    Yields:
        Path object to the temporary file
    """
    temp_file_obj = None
    temp_path = None
    
    try:
        temp_file_obj = tempfile.NamedTemporaryFile(
            suffix=suffix,
            prefix=prefix,
            dir=dir,
            delete=False  # We handle deletion manually
        )
        temp_path = Path(temp_file_obj.name)
        temp_file_obj.close()  # Close handle immediately, let user manage the file
        
        logger.debug(f"Created temporary file: {temp_path}")
        yield temp_path
        
    except Exception as e:
        logger.error(f"Error in temporary_file context: {e}")
        raise TemporaryFileError(
            f"Failed to create temporary file: {e}",
            details={"suffix": suffix, "prefix": prefix}
        )
    finally:
        # Clean up the temporary file if it exists and delete=True
        if delete and temp_path and temp_path.exists():
            try:
                temp_path.unlink()
                logger.debug(f"Deleted temporary file: {temp_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to delete temporary file {temp_path}: {cleanup_error}")


@contextmanager
def temporary_directory(
    suffix: str = "",
    prefix: str = "temp_dir_",
    dir: Optional[str] = None
) -> Generator[Path, None, None]:
    """
    Context manager for creating and managing temporary directories.
    
    Usage:
        with temporary_directory() as temp_dir:
            # Use temp_dir for file operations
            file_path = temp_dir / "data.txt"
            file_path.write_text("content")
        # Directory and all contents are automatically deleted
    
    Args:
        suffix: Directory name suffix
        prefix: Directory name prefix
        dir: Parent directory for temp dir (None = system temp)
        
    Yields:
        Path object to the temporary directory
    """
    temp_dir = None
    
    try:
        temp_dir = Path(tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir))
        logger.debug(f"Created temporary directory: {temp_dir}")
        yield temp_dir
        
    except Exception as e:
        logger.error(f"Error in temporary_directory context: {e}")
        raise TemporaryFileError(
            f"Failed to create temporary directory: {e}",
            details={"suffix": suffix, "prefix": prefix}
        )
    finally:
        # Clean up the temporary directory and all its contents
        if temp_dir and temp_dir.exists():
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logger.debug(f"Deleted temporary directory: {temp_dir}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to delete temporary directory {temp_dir}: {cleanup_error}")


# =============================================================================
# BATCH PROCESSING CONTEXT MANAGER
# =============================================================================

@contextmanager
def batch_processing(batch_name: str = "batch"):
    """
    Context manager for tracking batch processing operations.
    Useful for logging and resource management in batch operations.
    
    Usage:
        with batch_processing("video_frames"):
            for frame in frames:
                process_frame(frame)
        # Automatic logging of batch completion
    
    Args:
        batch_name: Name for this batch operation
    """
    import time
    start_time = time.time()
    logger.info(f"Starting batch processing: {batch_name}")
    
    try:
        yield
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Batch processing '{batch_name}' failed after {elapsed:.2f}s: {e}")
        raise
    else:
        elapsed = time.time() - start_time
        logger.info(f"Batch processing '{batch_name}' completed in {elapsed:.2f}s")


# =============================================================================
# TORCH INFERENCE CONTEXT MANAGER
# =============================================================================

@contextmanager
def torch_inference_mode():
    """
    Context manager for PyTorch inference that ensures proper mode and cleanup.
    
    Usage:
        with torch_inference_mode():
            output = model(input_tensor)
        # torch.no_grad() and model.eval() are automatically handled
    """
    import torch
    
    # Store original grad state (though we expect it to be disabled)
    grad_enabled = torch.is_grad_enabled()
    
    try:
        torch.set_grad_enabled(False)
        yield
    finally:
        # Restore original state
        torch.set_grad_enabled(grad_enabled)
