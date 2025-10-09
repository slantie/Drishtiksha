# src/ml/health_check.py

import time
import torch
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from PIL import Image

from src.ml.correlation import get_logger
from src.ml.registry import ModelManager
from src.ml.exceptions import ModelLoadError, InferenceError

logger = get_logger(__name__)


class ModelHealthStatus:
    """Represents the health status of a single model."""
    
    def __init__(
        self,
        model_name: str,
        is_healthy: bool,
        load_status: str,
        inference_status: Optional[str] = None,
        error_message: Optional[str] = None,
        warmup_time: Optional[float] = None,
        last_checked: Optional[datetime] = None
    ):
        self.model_name = model_name
        self.is_healthy = is_healthy
        self.load_status = load_status
        self.inference_status = inference_status
        self.error_message = error_message
        self.warmup_time = warmup_time
        self.last_checked = last_checked or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "model_name": self.model_name,
            "is_healthy": self.is_healthy,
            "load_status": self.load_status,
            "inference_status": self.inference_status,
            "error_message": self.error_message,
            "warmup_time": self.warmup_time,
            "last_checked": self.last_checked.isoformat()
        }


class HealthChecker:
    """
    Comprehensive health checker for all ML models.
    
    This class performs deep health checks including:
    - Model load verification
    - Inference capability validation
    - Resource availability checks
    - Performance benchmarking
    """
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self._health_cache: Dict[str, ModelHealthStatus] = {}
        self._cache_duration_seconds = 300  # Cache for 5 minutes
        
        # Test assets paths
        self.test_video_path = Path("assets/test_video.mp4")
        self.test_audio_path = Path("assets/test_audio.mp3")
        self.test_image_path = Path("assets/test_image.jpg")
    
    async def check_all_models(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive health check on all loaded models.
        
        Args:
            force_refresh: If True, bypass cache and perform fresh checks
            
        Returns:
            Dict containing health status for all models
        """
        logger.info("Starting comprehensive health check for all models")
        start_time = time.time()
        
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "models": [],
            "summary": {
                "total_models": 0,
                "healthy_models": 0,
                "unhealthy_models": 0,
                "check_duration": 0.0
            }
        }
        
        # Check each loaded model
        for model_name, model_instance in self.model_manager.loaded_models.items():
            if not force_refresh and model_name in self._health_cache:
                # Use cached result if still valid
                cached_status = self._health_cache[model_name]
                cache_age = (datetime.utcnow() - cached_status.last_checked).total_seconds()
                
                if cache_age < self._cache_duration_seconds:
                    logger.info(f"Using cached health status for {model_name} (age: {cache_age:.1f}s)")
                    results["models"].append(cached_status.to_dict())
                    if cached_status.is_healthy:
                        results["summary"]["healthy_models"] += 1
                    else:
                        results["summary"]["unhealthy_models"] += 1
                    continue
            
            # Perform fresh health check
            health_status = await self._check_single_model(model_name, model_instance)
            self._health_cache[model_name] = health_status
            
            results["models"].append(health_status.to_dict())
            if health_status.is_healthy:
                results["summary"]["healthy_models"] += 1
            else:
                results["summary"]["unhealthy_models"] += 1
        
        results["summary"]["total_models"] = len(results["models"])
        results["summary"]["check_duration"] = time.time() - start_time
        
        logger.info(
            f"Health check complete: {results['summary']['healthy_models']}/{results['summary']['total_models']} "
            f"models healthy in {results['summary']['check_duration']:.2f}s"
        )
        
        return results
    
    async def _check_single_model(self, model_name: str, model_instance: Any) -> ModelHealthStatus:
        """
        Perform deep health check on a single model.
        
        Args:
            model_name: Name of the model
            model_instance: The loaded model instance
            
        Returns:
            ModelHealthStatus object
        """
        logger.info(f"Checking health of model: {model_name}")
        
        try:
            # 1. Verify model is loaded
            if model_instance.model is None:
                return ModelHealthStatus(
                    model_name=model_name,
                    is_healthy=False,
                    load_status="NOT_LOADED",
                    error_message="Model object is None"
                )
            
            # 2. Verify model is on correct device
            try:
                if hasattr(model_instance.model, 'device'):
                    model_device = str(model_instance.model.device)
                elif next(model_instance.model.parameters(), None) is not None:
                    model_device = str(next(model_instance.model.parameters()).device)
                else:
                    model_device = model_instance.device
                
                logger.debug(f"Model {model_name} is on device: {model_device}")
            except Exception as e:
                logger.warning(f"Could not verify device for {model_name}: {e}")
            
            # 3. Perform inference test with synthetic data
            inference_result = await self._test_inference(model_name, model_instance)
            
            if inference_result["success"]:
                return ModelHealthStatus(
                    model_name=model_name,
                    is_healthy=True,
                    load_status="LOADED",
                    inference_status="OPERATIONAL",
                    warmup_time=inference_result.get("inference_time")
                )
            else:
                return ModelHealthStatus(
                    model_name=model_name,
                    is_healthy=False,
                    load_status="LOADED",
                    inference_status="INFERENCE_FAILED",
                    error_message=inference_result.get("error")
                )
        
        except Exception as e:
            logger.error(f"Health check failed for {model_name}: {e}", exc_info=True)
            return ModelHealthStatus(
                model_name=model_name,
                is_healthy=False,
                load_status="ERROR",
                error_message=str(e)
            )
    
    async def _test_inference(self, model_name: str, model_instance: Any) -> Dict[str, Any]:
        """
        Test if the model can perform inference successfully.
        
        Args:
            model_name: Name of the model
            model_instance: The loaded model instance
            
        Returns:
            Dict with success status and timing information
        """
        try:
            # Determine model type from config
            config = model_instance.config
            is_video = getattr(config, 'is_video', False)
            is_audio = getattr(config, 'is_audio', False)
            is_image = getattr(config, 'is_image', False)
            
            # Create appropriate synthetic test data
            if is_video:
                test_path = self._create_synthetic_video()
            elif is_audio:
                test_path = self._create_synthetic_audio()
            elif is_image:
                test_path = self._create_synthetic_image()
            else:
                # Fallback: try to determine from class name
                class_name = model_instance.config.class_name.lower()
                if 'audio' in class_name or 'scattering' in class_name or 'mel' in class_name or 'stft' in class_name:
                    test_path = self._create_synthetic_audio()
                else:
                    test_path = self._create_synthetic_video()
            
            logger.debug(f"Running inference test for {model_name} with synthetic data")
            start_time = time.time()
            
            # Run inference in thread pool to avoid blocking
            result = await asyncio.to_thread(
                model_instance.analyze,
                str(test_path)
            )
            
            inference_time = time.time() - start_time
            
            # Cleanup synthetic file
            if test_path.exists():
                test_path.unlink()
            
            # Verify result structure
            if result is None:
                return {"success": False, "error": "Model returned None"}
            
            if not hasattr(result, 'prediction'):
                return {"success": False, "error": "Result missing 'prediction' field"}
            
            logger.info(f"Inference test passed for {model_name} in {inference_time:.3f}s")
            return {
                "success": True,
                "inference_time": inference_time,
                "prediction": result.prediction
            }
        
        except Exception as e:
            logger.error(f"Inference test failed for {model_name}: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    def _create_synthetic_video(self) -> Path:
        """Create a minimal synthetic video for testing."""
        import cv2
        temp_path = Path("/tmp") / f"health_check_video_{time.time()}.mp4"
        
        # Create a simple 1-second video with 30 frames
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_path), fourcc, 30.0, (224, 224))
        
        for i in range(30):
            # Create a frame with random noise
            frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
        return temp_path
    
    def _create_synthetic_audio(self) -> Path:
        """Create a minimal synthetic audio file for testing."""
        import wave
        temp_path = Path("/tmp") / f"health_check_audio_{time.time()}.wav"
        
        # Create a 2-second audio file
        sample_rate = 16000
        duration = 2
        frequency = 440  # A4 note
        
        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
        
        # Write WAV file
        with wave.open(str(temp_path), 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        return temp_path
    
    def _create_synthetic_image(self) -> Path:
        """Create a minimal synthetic image for testing."""
        temp_path = Path("/tmp") / f"health_check_image_{time.time()}.jpg"
        
        # Create a random RGB image
        image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        image.save(temp_path, 'JPEG')
        
        return temp_path
    
    def get_cached_status(self, model_name: str) -> Optional[ModelHealthStatus]:
        """Get cached health status for a specific model."""
        return self._health_cache.get(model_name)
    
    def clear_cache(self) -> None:
        """Clear the health check cache."""
        logger.info("Clearing health check cache")
        self._health_cache.clear()
