# src/app/schemas.py

from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, TypeVar, Generic

# --- Standard Error and Response Wrappers ---

class APIError(BaseModel):
    """Standard model for API error responses."""
    error: str
    message: str
    details: Optional[Union[str, Dict[str, Any]]] = None

DataType = TypeVar('DataType')

class APIResponse(BaseModel, Generic[DataType]):
    """Standard wrapper for all successful API responses."""
    success: bool = True
    model_used: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: DataType

# --- Health and Status Schemas (Unchanged) ---

class ModelStatus(BaseModel):
    name: str
    loaded: bool
    description: str

class HealthStatus(BaseModel):
    """Detailed health status of the service."""
    status: str
    active_models: List[ModelStatus]
    default_model: str

class DeviceInfo(BaseModel):
    """Information about the compute device."""
    type: str
    name: Optional[str] = None
    total_memory: Optional[float] = None
    used_memory: Optional[float] = None
    free_memory: Optional[float] = None
    memory_usage_percent: Optional[float] = None
    compute_capability: Optional[str] = None
    cuda_version: Optional[str] = None

class SystemInfo(BaseModel):
    """System resource information."""
    python_version: str
    platform: str
    cpu_count: int
    total_ram: float
    used_ram: float
    ram_usage_percent: float
    uptime_seconds: float

class ModelInfo(BaseModel):
    """Detailed information about a loaded model."""
    name: str
    class_name: str
    description: str
    loaded: bool
    device: str
    model_path: str
    isAudio: bool = False
    isVideo: bool = False
    memory_usage_mb: Optional[float] = None
    load_time: Optional[float] = None
    inference_count: Optional[int] = 0

class ServerStats(BaseModel):
    """Comprehensive server statistics."""
    service_name: str
    version: str
    status: str
    uptime_seconds: float
    device_info: DeviceInfo
    system_info: SystemInfo
    models_info: List[ModelInfo]
    active_models_count: int
    total_models_count: int
    configuration: Dict[str, Any]

# --- Unified Analysis Schemas ---

class BaseAnalysisResult(BaseModel):
    """
    The base schema for all model analysis results. Contains fields
    that are common across all media types.
    """
    prediction: str = Field(..., description="The final prediction ('REAL' or 'FAKE').")
    confidence: float = Field(..., description="The model's confidence in the prediction (0.0 to 1.0).")
    processing_time: float = Field(..., description="Total time taken for the analysis in seconds.")
    note: Optional[str] = Field(None, description="An optional note about the analysis, e.g., for fallbacks or warnings.")

class FramePrediction(BaseModel):
    """Represents the analysis result for a single frame or sequence window."""
    index: int = Field(..., description="The index of the frame or window.")
    score: float = Field(..., description="The raw 'fake' score for this frame/window (0.0 to 1.0).")
    prediction: str = Field(..., description="The prediction for this frame/window ('REAL' or 'FAKE').")

class VideoAnalysisResult(BaseAnalysisResult):
    """
    A comprehensive data structure for video analysis results.
    This single schema will be the return type for all video models.
    """
    media_type: str = "video"
    frame_count: Optional[int] = Field(None, description="Total frames in the video.")
    frames_analyzed: int = Field(..., description="Number of frames or windows analyzed.")
    frame_predictions: List[FramePrediction] = Field([], description="A list of predictions for each analyzed frame or window.")
    metrics: Dict[str, Any] = Field({}, description="A dictionary of additional, model-specific metrics (e.g., average scores, variance).")
    visualization_path: Optional[str] = Field(None, description="An optional path to a generated visualization video file.")

# --- Audio Schemas (Now inheriting from BaseAnalysisResult) ---

class PitchAnalysis(BaseModel):
    mean_pitch_hz: Optional[float] = None
    pitch_stability_score: Optional[float] = None

class EnergyAnalysis(BaseModel):
    rms_energy: float
    silence_ratio: float

class SpectralAnalysis(BaseModel):
    spectral_centroid: float
    spectral_contrast: float

class AudioProperties(BaseModel):
    duration_seconds: float
    sample_rate: int
    channels: int

class AudioVisualization(BaseModel):
    spectrogram_url: Optional[str] = None
    spectrogram_data: Optional[List[List[float]]] = None

class AudioAnalysisResult(BaseAnalysisResult):
    """

    A comprehensive data structure for audio analysis results.
    Inherits common fields and adds audio-specific ones.
    """
    media_type: str = "audio"
    properties: AudioProperties
    pitch: PitchAnalysis
    energy: EnergyAnalysis
    spectral: SpectralAnalysis
    visualization: Optional[AudioVisualization] = None


# --- LEGACY SCHEMAS (To be deprecated and removed later) ---

class AnalysisData(BaseModel):
    prediction: str
    confidence: float
    processing_time: float
    metrics: Optional[Dict[str, Any]] = None
    note: Optional[str] = None

class FramesAnalysisData(BaseModel):
    overall_prediction: str
    overall_confidence: float
    processing_time: float
    frame_predictions: List[FramePrediction] # Adjusted to use the new FramePrediction
    temporal_analysis: Dict[str, Any]
    note: Optional[str] = None

class ComprehensiveAnalysisData(AnalysisData):
    frames_analysis: Optional[FramesAnalysisData] = None
    visualization_generated: bool = False
    visualization_filename: Optional[str] = None
    processing_breakdown: Dict[str, float] = {}

class AudioAnalysisData(BaseModel):
    prediction: str
    confidence: float
    processing_time: float
    note: Optional[str] = None
    properties: AudioProperties
    pitch: PitchAnalysis
    energy: EnergyAnalysis
    spectral: SpectralAnalysis
    visualization: AudioVisualization
    
class ImageAnalysisResult(BaseAnalysisResult):
    """
    A data structure for image analysis results.
    """
    media_type: str = "image"
    dimensions: Dict[str, int] = Field(..., description="The width and height of the analyzed image.")
    heatmap_scores: Optional[List[List[float]]] = Field(None, description="A 2D grid of suspicion scores from patch-based analysis.")

AnalysisResult = Union[VideoAnalysisResult, AudioAnalysisResult, ImageAnalysisResult]