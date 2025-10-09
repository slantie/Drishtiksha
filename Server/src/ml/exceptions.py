# src/ml/exceptions.py

"""
Comprehensive exception hierarchy for ML models.
This provides clear, specific error types for different failure modes.
"""

# =============================================================================
# BASE EXCEPTIONS
# =============================================================================

class ModelError(Exception):
    """
    Base exception for all custom errors raised by the models.
    All ML-related exceptions should inherit from this.
    """
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


# =============================================================================
# MEDIA PROCESSING EXCEPTIONS (Client-side errors - 4xx)
# =============================================================================

class MediaProcessingError(ModelError):
    """
    Base class for all media processing errors.
    These are typically user-fixable issues (4xx HTTP status).
    """
    pass


class UnsupportedMediaFormatError(MediaProcessingError):
    """Media file format is not supported by the model."""
    pass


class CorruptedMediaError(MediaProcessingError):
    """Media file is corrupted or cannot be decoded."""
    pass


class NoFacesDetectedError(MediaProcessingError):
    """No faces detected in the media (for face-based models)."""
    pass


class NoAudioTrackError(MediaProcessingError):
    """Video file has no audio track (for audio models)."""
    pass


class InsufficientFramesError(MediaProcessingError):
    """Video has too few frames for meaningful analysis."""
    pass


class InvalidMediaDurationError(MediaProcessingError):
    """Media duration is outside acceptable range."""
    pass


# =============================================================================
# MODEL/INFERENCE EXCEPTIONS (Server-side errors - 5xx)
# =============================================================================

class InferenceError(ModelError):
    """
    Base class for errors during model inference.
    These are typically server/configuration issues (5xx HTTP status).
    """
    pass


class ModelNotLoadedError(InferenceError):
    """Model has not been loaded into memory."""
    pass


class GPUOutOfMemoryError(InferenceError):
    """GPU ran out of memory during inference."""
    pass


class ModelInferenceTimeoutError(InferenceError):
    """Model inference exceeded maximum allowed time."""
    pass


class PreprocessingError(InferenceError):
    """Error during preprocessing of media for inference."""
    pass


class PostprocessingError(InferenceError):
    """Error during postprocessing of model outputs."""
    pass


# =============================================================================
# RESOURCE EXCEPTIONS
# =============================================================================

class ResourceError(ModelError):
    """Base class for resource-related errors."""
    pass


class TemporaryFileError(ResourceError):
    """Error creating or managing temporary files."""
    pass


class VideoCapturError(ResourceError):
    """Error opening or reading from video capture device."""
    pass


class VisualizationError(ResourceError):
    """Error generating visualization artifacts."""
    pass


# =============================================================================
# CONFIGURATION EXCEPTIONS
# =============================================================================

class ConfigurationError(ModelError):
    """Base class for configuration-related errors."""
    pass


class ModelWeightsNotFoundError(ConfigurationError):
    """Model weights file not found at specified path."""
    pass


class InvalidConfigurationError(ConfigurationError):
    """Model configuration is invalid or incomplete."""
    pass


class ModelLoadError(ConfigurationError):
    """Model failed to load properly."""
    pass


class ModelNotFoundError(ConfigurationError):
    """Requested model not found in registry."""
    pass


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def wrap_external_exception(exc: Exception, context: str = "") -> ModelError:
    """
    Wraps external exceptions (OpenCV, PyTorch, etc.) into our exception hierarchy.
    
    Args:
        exc: The external exception to wrap
        context: Additional context about what was being done
        
    Returns:
        An appropriate ModelError subclass
    """
    error_msg = f"{context}: {str(exc)}" if context else str(exc)
    
    # Map common external exceptions to our hierarchy
    if isinstance(exc, FileNotFoundError):
        return ModelWeightsNotFoundError(error_msg, {"original_exception": type(exc).__name__})
    
    if isinstance(exc, (IOError, OSError)):
        return MediaProcessingError(error_msg, {"original_exception": type(exc).__name__})
    
    if "CUDA out of memory" in str(exc):
        return GPUOutOfMemoryError(error_msg, {"original_exception": type(exc).__name__})
    
    if "timeout" in str(exc).lower():
        return ModelInferenceTimeoutError(error_msg, {"original_exception": type(exc).__name__})
    
    # Default: wrap as generic InferenceError
    return InferenceError(error_msg, {"original_exception": type(exc).__name__})