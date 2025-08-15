# src/app/schemas.py

from pydantic import BaseModel
from typing import List


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


class DetailedMetrics(BaseModel):
    frame_count: int
    per_frame_scores: List[float]
    rolling_average_scores: List[float]
    final_average_score: float
    max_score: float
    min_score: float
    score_variance: float
    suspicious_frames_count: int
    suspicious_frames_percentage: float


class DetailedAnalysisResult(BaseModel):
    prediction: str
    confidence: float
    processing_time: float
    model_version: str
    metrics: DetailedMetrics


class DetailedAnalysisResponse(BaseModel):
    success: bool = True
    video_id: str
    result: DetailedAnalysisResult


class FrameAnalysisSummary(BaseModel):
    prediction: str
    confidence: float
    frame_count: int
    suspicious_frames: int
    average_suspicion: float
    max_suspicion: float
    min_suspicion: float
    suspicion_variance: float


class FrameAnalysisResponse(BaseModel):
    success: bool = True
    video_id: str
    summary: FrameAnalysisSummary


class ColorCuesTemporalAnalysis(BaseModel):
    per_sequence_scores: List[float]
    rolling_averages: List[float]
    sequence_count: int
    avg_score: float
    max_score: float
    min_score: float
    score_variance: float
    suspicious_sequences: int
    suspicious_percentage: float


class ColorCuesAnalysisResult(BaseModel):
    prediction: str
    confidence: float
    processing_time: float
    fake_probability: float
    max_fake_score: float
    min_fake_score: float
    score_variance: float
    num_sequences: int
    num_features_extracted: int
    model_version: str


class ColorCuesDetailedResult(BaseModel):
    prediction: str
    confidence: float
    processing_time: float
    temporal_analysis: ColorCuesTemporalAnalysis
    model_version: str


class ColorCuesAnalysisResponse(BaseModel):
    success: bool = True
    video_id: str
    result: ColorCuesAnalysisResult


class ColorCuesDetailedResponse(BaseModel):
    success: bool = True
    video_id: str
    result: ColorCuesDetailedResult
