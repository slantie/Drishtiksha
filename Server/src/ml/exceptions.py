# src/ml/exceptions.py

class ModelError(Exception):
    """Base exception for all custom errors raised by the models."""
    pass

class MediaProcessingError(ModelError):
    """
    Raised when a media file (video or audio) cannot be processed due to corruption,
    unsupported format, or other content-related issues.
    
    This typically corresponds to a client-side error (4xx HTTP status).
    """
    pass

class InferenceError(ModelError):
    """
    Raised when an error occurs during the model inference/prediction phase.
    
    This could be due to issues like GPU out-of-memory errors and typically
    corresponds to a server-side error (5xx HTTP status).
    """
    pass