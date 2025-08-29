# src/ml/schemas.py

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal

# The type of event being published.
EventType = Literal[
    "FRAME_ANALYSIS_PROGRESS",
    "AUDIO_EXTRACTION_START",
    "AUDIO_EXTRACTION_COMPLETE",
    "SPECTROGRAM_GENERATION_START",
    "SPECTROGRAM_GENERATION_COMPLETE",
    "ANALYSIS_COMPLETE",
    "ANALYSIS_FAILED",
]

class EventData(BaseModel):
    """
    A flexible data payload for events. Can contain any relevant
    information, such as progress percentage or model name.
    """
    model_name: str
    progress: Optional[int] = None
    total: Optional[int] = None
    details: Optional[Dict[str, Any]] = None

class ProgressEvent(BaseModel):
    """
    The main schema for all progress events published to Redis.
    This enforces a consistent structure for all event messages.
    """
    media_id: str = Field(..., description="The unique identifier for the media being processed (e.g., videoId).")
    user_id: Optional[str] = Field(None, description="The identifier for the user who initiated the request.")
    event: EventType = Field(..., description="The specific event type that occurred.")
    message: str = Field(..., description="A human-readable message describing the event.")
    data: EventData