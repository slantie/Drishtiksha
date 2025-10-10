# src/ml/warmup.py

"""
Model warmup functionality to eliminate cold-start latency.
Runs a dummy inference on each model after loading to ensure all
components are initialized and cached.
"""

import logging
import time
import numpy as np
import torch
from pathlib import Path
from typing import Optional

from src.ml.exceptions import InferenceError
from src.ml.context_managers import temporary_file

logger = logging.getLogger(__name__)


def create_dummy_video(duration_seconds: float = 2.0, fps: int = 30, resolution: tuple[int, int] = (224, 224)) -> Path:
    """
    Creates a dummy video file for warmup purposes.
    
    Args:
        duration_seconds: Duration of the video
        fps: Frames per second
        resolution: (width, height) of the video
        
    Returns:
        Path to the created dummy video
    """
    import cv2
    from src.ml.context_managers import temporary_file, video_writer
    
    try:
        # Create a temporary file that we'll keep for warmup
        temp_video = Path("temp/warmup_video.mp4")
        temp_video.parent.mkdir(parents=True, exist_ok=True)
        
        # If warmup video already exists, reuse it
        if temp_video.exists():
            logger.debug(f"Reusing existing warmup video: {temp_video}")
            return temp_video
        
        # Generate random frames
        num_frames = int(duration_seconds * fps)
        width, height = resolution
        
        with video_writer(str(temp_video), "mp4v", fps, (width, height)) as writer:
            for i in range(num_frames):
                # Generate a frame with random noise
                frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                writer.write(frame)
        
        logger.info(f"Created warmup video: {temp_video} ({num_frames} frames, {duration_seconds}s)")
        return temp_video
        
    except Exception as e:
        logger.error(f"Failed to create dummy video for warmup: {e}")
        raise


def create_dummy_audio(duration_seconds: float = 2.0, sample_rate: int = 16000) -> Path:
    """
    Creates a dummy audio file for warmup purposes.
    
    Args:
        duration_seconds: Duration of the audio
        sample_rate: Audio sample rate
        
    Returns:
        Path to the created dummy audio
    """
    import soundfile as sf
    
    try:
        # Create a temporary file that we'll keep for warmup
        temp_audio = Path("temp/warmup_audio.wav")
        temp_audio.parent.mkdir(parents=True, exist_ok=True)
        
        # If warmup audio already exists, reuse it
        if temp_audio.exists():
            logger.debug(f"Reusing existing warmup audio: {temp_audio}")
            return temp_audio
        
        # Generate random audio samples
        num_samples = int(duration_seconds * sample_rate)
        audio_data = np.random.randn(num_samples).astype(np.float32) * 0.1  # Low amplitude noise
        
        # Write to file
        sf.write(str(temp_audio), audio_data, sample_rate)
        
        logger.info(f"Created warmup audio: {temp_audio} ({duration_seconds}s, {sample_rate}Hz)")
        return temp_audio
        
    except Exception as e:
        logger.error(f"Failed to create dummy audio for warmup: {e}")
        raise


def create_dummy_image(resolution: tuple[int, int] = (224, 224)) -> Path:
    """
    Creates a dummy image file for warmup purposes.
    
    Args:
        resolution: (width, height) of the image
        
    Returns:
        Path to the created dummy image
    """
    from PIL import Image
    
    try:
        # Create a temporary file that we'll keep for warmup
        temp_image = Path("temp/warmup_image.jpg")
        temp_image.parent.mkdir(parents=True, exist_ok=True)
        
        # If warmup image already exists, reuse it
        if temp_image.exists():
            logger.debug(f"Reusing existing warmup image: {temp_image}")
            return temp_image
        
        # Generate random image
        width, height = resolution
        image_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        image.save(temp_image)
        
        logger.info(f"Created warmup image: {temp_image} ({width}x{height})")
        return temp_image
        
    except Exception as e:
        logger.error(f"Failed to create dummy image for warmup: {e}")
        raise


def warmup_model(model: "BaseModel", model_type: str = "video") -> tuple[bool, float, Optional[str]]:
    """
    Performs a warmup inference on a model to eliminate cold-start latency.
    
    Args:
        model: The model instance to warm up
        model_type: Type of model ("video", "audio", or "image")
        
    Returns:
        Tuple of (success, inference_time, error_message)
    """
    try:
        logger.info(f"Warming up model: {model.config.model_name} ({model_type})")
        start_time = time.time()
        
        # Create appropriate dummy media based on model type
        if model_type == "video":
            dummy_media = create_dummy_video()
        elif model_type == "audio":
            dummy_media = create_dummy_audio()
        elif model_type == "image":
            dummy_media = create_dummy_image()
        else:
            logger.warning(f"Unknown model type for warmup: {model_type}")
            return False, 0.0, f"Unknown model type: {model_type}"
        
        # Run inference
        result = model.analyze(str(dummy_media))
        
        inference_time = time.time() - start_time
        logger.info(
            f"✅ Warmup successful for {model.config.model_name}: "
            f"{inference_time:.2f}s (prediction: {result.prediction})"
        )
        
        return True, inference_time, None
        
    except Exception as e:
        inference_time = time.time() - start_time
        error_msg = f"Warmup failed: {str(e)}"
        logger.error(f"❌ {error_msg} for {model.config.model_name} after {inference_time:.2f}s", exc_info=True)
        return False, inference_time, error_msg


def warmup_all_models(model_manager: "ModelManager") -> dict[str, dict]:
    """
    Warms up all loaded models in the model manager.
    
    Args:
        model_manager: The ModelManager instance containing loaded models
        
    Returns:
        Dictionary mapping model names to warmup results:
        {
            "model_name": {
                "success": bool,
                "inference_time": float,
                "error": Optional[str]
            }
        }
    """
    logger.info("=" * 70)
    logger.info("STARTING MODEL WARMUP PHASE")
    logger.info("=" * 70)
    
    warmup_results = {}
    total_start = time.time()
    
    for model_name, model in model_manager._models.items():
        # Determine model type from config or class name
        model_type = "video"  # Default
        
        # Try to infer type from configuration flags
        if hasattr(model.config, "isAudio") and model.config.isAudio:
            model_type = "audio"
        elif hasattr(model.config, "isImage") and model.config.isImage:
            model_type = "image"
        elif hasattr(model.config, "isVideo") and model.config.isVideo:
            model_type = "video"
        
        success, inference_time, error = warmup_model(model, model_type)
        
        warmup_results[model_name] = {
            "success": success,
            "inference_time": inference_time,
            "error": error
        }
    
    total_time = time.time() - total_start
    successful = sum(1 for r in warmup_results.values() if r["success"])
    failed = len(warmup_results) - successful
    
    logger.info("=" * 70)
    logger.info(f"WARMUP COMPLETE: {successful} successful, {failed} failed, {total_time:.2f}s total")
    logger.info("=" * 70)
    
    if failed > 0:
        logger.warning(f"⚠️  {failed} model(s) failed warmup. Check logs for details.")
    
    return warmup_results
