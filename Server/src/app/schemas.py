# src/app/schemas.py

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, TypeVar, Generic
from datetime import datetime

# --- Standard Error and Response Wrappers ---

class APIError(BaseModel):
    """Standard model for API error responses."""
    error: str
    message: str
    details: Optional[Union[str, Dict[str, Any]]] = None

DataType = TypeVar('DataType')

# --- REFACTORED: Make APIResponse a Generic class ---
class APIResponse(BaseModel, Generic[DataType]):
    """Standard wrapper for all successful API responses."""
    success: bool = True
    model_used: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: DataType

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

# --- Server Statistics Schemas ---

class DeviceInfo(BaseModel):
    """Information about the compute device."""
    type: str  # "cuda" or "cpu"
    name: Optional[str] = None
    total_memory: Optional[float] = None  # In GB
    used_memory: Optional[float] = None   # In GB
    free_memory: Optional[float] = None   # In GB
    memory_usage_percent: Optional[float] = None
    compute_capability: Optional[str] = None
    cuda_version: Optional[str] = None

class SystemInfo(BaseModel):
    """System resource information."""
    python_version: str
    platform: str
    cpu_count: int
    total_ram: float  # In GB
    used_ram: float   # In GB
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
    isDetailed: bool
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

# --- Analysis Schemas (One for each endpoint type) ---

class AnalysisData(BaseModel):
    """Data model for the unified /analyze endpoint."""
    prediction: str
    confidence: float
    processing_time: float
    metrics: Optional[Dict[str, Any]] = None  # Includes detailed metrics if available
    note: Optional[str] = None

# Legacy schemas kept for backward compatibility
class QuickAnalysisData(AnalysisData):
    """Data model for backward compatibility."""
    pass

class DetailedAnalysisData(AnalysisData):
    """Data model for backward compatibility."""
    metrics: Dict[str, Any]  # Required for detailed analysis

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

class ComprehensiveAnalysisData(BaseModel):
    """Data model for the merged comprehensive analysis endpoint."""
    # Basic analysis results
    prediction: str
    confidence: float
    processing_time: float
    metrics: Optional[Dict[str, Any]] = None
    note: Optional[str] = None
    
    # Frame-by-frame analysis results
    frames_analysis: Optional[FramesAnalysisData] = None
    
    # Visualization info
    visualization_generated: bool = False
    visualization_filename: Optional[str] = None
    
    # Processing breakdown
    processing_breakdown: Dict[str, float] = {}

class PitchAnalysis(BaseModel):
    mean_pitch_hz: Optional[float] = Field(None, description="Average fundamental frequency (pitch) of the audio in Hertz.")
    pitch_stability_score: Optional[float] = Field(None, description="A score from 0.0 to 1.0 indicating how stable the pitch is. Unnaturally flat or erratic pitch results in a lower score.")

class EnergyAnalysis(BaseModel):
    rms_energy: float = Field(..., description="Root Mean Square energy, indicating the audio's loudness.")
    silence_ratio: float = Field(..., description="The proportion of the audio clip that is considered silent (below a threshold).")

class SpectralAnalysis(BaseModel):
    spectral_centroid: float = Field(..., description="The 'center of mass' of the spectrum, indicating brightness.")
    spectral_contrast: float = Field(..., description="The difference between spectral peaks and valleys. Can indicate clarity.")

class AudioProperties(BaseModel):
    duration_seconds: float
    sample_rate: int
    channels: int

class AudioVisualization(BaseModel):
    spectrogram_url: str = Field(..., description="A temporary URL to download the generated Mel Spectrogram image.")
    spectrogram_data: List[List[float]] = Field(..., description="The raw Mel Spectrogram data (dB-scaled). Format: [frequency_bins][time_steps]. Ready for charting.")

class AudioAnalysisData(BaseModel):
    """Data model for a comprehensive audio analysis response."""
    prediction: str
    confidence: float
    processing_time: float
    note: Optional[str] = None
    
    properties: AudioProperties
    pitch: PitchAnalysis
    energy: EnergyAnalysis
    spectral: SpectralAnalysis
    visualization: AudioVisualization