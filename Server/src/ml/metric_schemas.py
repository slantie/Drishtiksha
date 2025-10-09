# src/ml/metric_schemas.py

"""
Strict metric schemas for different model types and modalities.
This ensures consistent response structure across all models.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


# =============================================================================
# VIDEO MODEL METRICS
# =============================================================================

class TemporalConsistencyMetrics(BaseModel):
    """Metrics related to temporal consistency analysis."""
    score_variance: float = Field(..., description="Variance in frame-level scores")
    score_std_dev: float = Field(..., description="Standard deviation of scores")
    consistency_score: float = Field(..., description="Overall temporal consistency (0-1)")
    suspicious_transitions: int = Field(0, description="Number of suspicious score transitions")


class FrameBasedMetrics(BaseModel):
    """Metrics for models that analyze individual frames."""
    total_frames: int = Field(..., description="Total frames in video")
    frames_analyzed: int = Field(..., description="Number of frames actually analyzed")
    sampling_strategy: str = Field(..., description="How frames were sampled (e.g., 'uniform', 'keyframes')")
    average_score: float = Field(..., description="Mean fake probability across all frames")
    max_score: float = Field(..., description="Maximum fake probability detected")
    min_score: float = Field(..., description="Minimum fake probability detected")
    temporal: TemporalConsistencyMetrics


class SequenceBasedMetrics(BaseModel):
    """Metrics for models that analyze frame sequences (LSTM-based)."""
    total_frames: int = Field(..., description="Total frames in video")
    num_sequences: int = Field(..., description="Number of sequences analyzed")
    sequence_length: int = Field(..., description="Frames per sequence")
    overlap_frames: int = Field(0, description="Overlapping frames between sequences")
    average_sequence_score: float = Field(..., description="Mean score across sequences")
    temporal: TemporalConsistencyMetrics


class FaceDetectionMetrics(BaseModel):
    """Metrics for face detection and per-face analysis."""
    frames_with_faces: int = Field(..., description="Frames where at least one face was detected")
    frames_without_faces: int = Field(..., description="Frames with no face detection")
    total_faces_detected: int = Field(..., description="Total face instances across all frames")
    average_faces_per_frame: float = Field(..., description="Average number of faces per frame")
    face_detection_confidence: float = Field(..., description="Average detection confidence")


class ColorCuesMetrics(BaseModel):
    """Specific metrics for color-based analysis."""
    chromaticity_variance: float = Field(..., description="Variance in facial chromaticity")
    histogram_consistency: float = Field(..., description="Consistency score for color histograms")
    lighting_variation: float = Field(..., description="Detected lighting variation")
    

class BlinkDetectionMetrics(BaseModel):
    """Specific metrics for eye blink analysis."""
    total_blinks_detected: int = Field(..., description="Number of blinks detected")
    average_blink_duration: float = Field(..., description="Average blink duration in seconds")
    blink_rate_per_minute: float = Field(..., description="Blinks per minute")
    natural_blink_score: float = Field(..., description="How natural the blink pattern is (0-1)")
    eye_aspect_ratio_variance: float = Field(..., description="Variance in EAR across video")


# =============================================================================
# AUDIO MODEL METRICS
# =============================================================================

class PitchMetrics(BaseModel):
    """Detailed pitch analysis metrics."""
    mean_pitch_hz: Optional[float] = Field(None, description="Average fundamental frequency")
    pitch_range_hz: Optional[float] = Field(None, description="Range between min and max pitch")
    pitch_stability_score: Optional[float] = Field(None, description="Consistency of pitch (0-1)")
    pitch_variance: Optional[float] = Field(None, description="Variance in pitch values")
    voiced_frames_ratio: Optional[float] = Field(None, description="Ratio of frames with detectable pitch")


class EnergyMetrics(BaseModel):
    """Audio energy and amplitude metrics."""
    rms_energy: float = Field(..., description="Root Mean Square energy")
    peak_energy: float = Field(..., description="Peak energy value")
    dynamic_range_db: float = Field(..., description="Dynamic range in decibels")
    silence_ratio: float = Field(..., description="Proportion of silent/low-energy frames")
    zero_crossing_rate: float = Field(..., description="Average zero-crossing rate")


class SpectralMetrics(BaseModel):
    """Frequency domain analysis metrics."""
    spectral_centroid: float = Field(..., description="Center of mass of spectrum")
    spectral_contrast: float = Field(..., description="Difference between peaks and valleys")
    spectral_rolloff: float = Field(..., description="Frequency below which 85% of energy lies")
    spectral_flatness: float = Field(..., description="Measure of spectral noisiness (0-1)")
    mfcc_features: Optional[List[float]] = Field(None, description="Mean MFCC coefficients")


class VoiceQualityMetrics(BaseModel):
    """Voice quality and naturalness metrics."""
    harmonicity: float = Field(..., description="Harmonics-to-noise ratio")
    jitter: float = Field(..., description="Pitch period variation")
    shimmer: float = Field(..., description="Amplitude variation")
    naturalness_score: float = Field(..., description="Overall voice naturalness (0-1)")


# =============================================================================
# IMAGE MODEL METRICS (for future use)
# =============================================================================

class ImageQualityMetrics(BaseModel):
    """Image quality and artifact detection metrics."""
    compression_quality_score: float = Field(..., description="Detected compression quality")
    noise_level: float = Field(..., description="Estimated noise level")
    blur_score: float = Field(..., description="Amount of blur detected (0-1)")
    edge_sharpness: float = Field(..., description="Edge sharpness metric")


class GanArtifactMetrics(BaseModel):
    """Metrics specific to GAN-generated image detection."""
    frequency_anomaly_score: float = Field(..., description="Anomalies in frequency domain")
    checkerboard_artifact_score: float = Field(..., description="Checkerboard pattern detection")
    color_bleeding_score: float = Field(..., description="Color bleeding artifacts")
    symmetry_anomaly_score: float = Field(..., description="Unnatural symmetry patterns")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_temporal_metrics(scores: List[float]) -> TemporalConsistencyMetrics:
    """Helper to create temporal consistency metrics from a list of scores."""
    import numpy as np
    
    scores_array = np.array(scores)
    variance = float(np.var(scores_array))
    std_dev = float(np.std(scores_array))
    
    # Calculate suspicious transitions (large jumps)
    if len(scores) > 1:
        diffs = np.abs(np.diff(scores_array))
        suspicious = int(np.sum(diffs > 0.3))  # Threshold for suspicious jump
    else:
        suspicious = 0
    
    # Consistency score: higher when variance is low
    consistency = float(1.0 - min(variance * 2, 1.0))
    
    return TemporalConsistencyMetrics(
        score_variance=variance,
        score_std_dev=std_dev,
        consistency_score=consistency,
        suspicious_transitions=suspicious
    )
