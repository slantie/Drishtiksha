# src/app/schemas.py

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

# --- Standard Error and Response Wrappers ---

class APIError(BaseModel):
    """Standard model for API error responses."""
    error: str
    message: str
    details: Optional[Union[str, Dict[str, Any]]] = None

class APIResponse(BaseModel):
    """Standard wrapper for all successful API responses."""
    success: bool = True
    model_used: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Any # The actual response data will go here

# --- Health and Status Schemas ---

class ModelStatus(BaseModel):
    name: str
    loaded: bool
    description: str

class HealthStatus(BaseModel):
    """Detailed health status of the service."""
    status: str
    active_models: List[ModelStatus]
    default_model: str

# --- Analysis Schemas (One for each endpoint type) ---

class QuickAnalysisData(BaseModel):
    """Data model for the /analyze endpoint."""
    prediction: str
    confidence: float
    processing_time: float
    note: Optional[str] = None

class DetailedAnalysisData(BaseModel):
    """
    Data model for the /analyze/detailed endpoint.
    The 'metrics' field is a flexible dictionary to accommodate different models.
    """
    prediction: str
    confidence: float
    processing_time: float
    metrics: Dict[str, Any]
    note: Optional[str] = None

class FramePrediction(BaseModel):
    """Represents the analysis result for a single frame."""
    frame_index: int
    score: float
    prediction: str

class FramesAnalysisData(BaseModel):
    """Data model for the /analyze/frames endpoint."""
    overall_prediction: str
    overall_confidence: float
    processing_time: float
    frame_predictions: List[FramePrediction]
    temporal_analysis: Dict[str, Any]
    note: Optional[str] = None