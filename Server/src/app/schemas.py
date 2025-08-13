# src/app/schemas.py

from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    default_model: str

class ModelInfo(BaseModel):
    model_name: str
    description: str
    model_path: str
    processor_path: str
    device: str

class ModelInfoResponse(BaseModel):
    success: bool = True
    model_info: ModelInfo

class AnalysisResult(BaseModel):
    prediction: str
    confidence: float
    processing_time: float
    model_version: str

class AnalysisResponse(BaseModel):
    success: bool = True
    video_id: str
    result: AnalysisResult