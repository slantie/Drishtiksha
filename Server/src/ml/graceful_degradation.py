# src/ml/graceful_degradation.py

import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.ml.correlation import get_logger
from src.ml.registry import ModelManager
from src.ml.exceptions import InferenceError, MediaProcessingError, ModelNotFoundError

logger = get_logger(__name__)


@dataclass
class ModelFailure:
    """Records a model failure for tracking and fallback decisions."""
    model_name: str
    error_type: str
    error_message: str
    timestamp: datetime
    media_path: Optional[str] = None


class FallbackStrategy:
    """
    Implements graceful degradation with intelligent fallback mechanisms.
    
    This class handles:
    - Model failure tracking
    - Automatic fallback to alternative models
    - Partial result aggregation
    - Circuit breaker pattern for failing models
    """
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self._failure_history: List[ModelFailure] = []
        self._failure_threshold = 3  # Number of failures before circuit opens
        self._circuit_reset_time = timedelta(minutes=10)  # Time before retry
        self._circuit_breaker: Dict[str, datetime] = {}  # model_name -> when circuit opened
    
    def is_model_available(self, model_name: str) -> bool:
        """
        Check if a model is currently available (circuit is closed).
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model can be used, False if circuit is open
        """
        if model_name not in self._circuit_breaker:
            return True
        
        # Check if enough time has passed to retry
        circuit_opened_at = self._circuit_breaker[model_name]
        if datetime.utcnow() - circuit_opened_at > self._circuit_reset_time:
            logger.info(f"Circuit breaker reset for model {model_name} after {self._circuit_reset_time}")
            del self._circuit_breaker[model_name]
            return True
        
        logger.warning(f"Circuit breaker OPEN for model {model_name}")
        return False
    
    def record_failure(self, model_name: str, error: Exception, media_path: Optional[str] = None) -> None:
        """
        Record a model failure and potentially open circuit breaker.
        
        Args:
            model_name: Name of the failed model
            error: The exception that occurred
            media_path: Optional path to the media that caused the failure
        """
        failure = ModelFailure(
            model_name=model_name,
            error_type=type(error).__name__,
            error_message=str(error),
            timestamp=datetime.utcnow(),
            media_path=media_path
        )
        
        self._failure_history.append(failure)
        logger.warning(f"Recorded failure for model {model_name}: {failure.error_type} - {failure.error_message}")
        
        # Check if we should open the circuit breaker
        recent_failures = self._get_recent_failures(model_name, within_minutes=5)
        
        if len(recent_failures) >= self._failure_threshold:
            if model_name not in self._circuit_breaker:
                self._circuit_breaker[model_name] = datetime.utcnow()
                logger.error(
                    f"🔴 CIRCUIT BREAKER OPENED for model {model_name} after {len(recent_failures)} "
                    f"failures in 5 minutes. Model will be unavailable for {self._circuit_reset_time}."
                )
    
    def record_success(self, model_name: str) -> None:
        """
        Record a successful inference. If circuit is open, this may close it.
        
        Args:
            model_name: Name of the model that succeeded
        """
        if model_name in self._circuit_breaker:
            logger.info(f"✅ Model {model_name} succeeded. Closing circuit breaker.")
            del self._circuit_breaker[model_name]
    
    def get_fallback_models(self, original_model: str, media_type: str) -> List[str]:
        """
        Get a list of fallback models for the same media type.
        
        Args:
            original_model: The model that failed
            media_type: Type of media ('video', 'audio', 'image')
            
        Returns:
            List of alternative model names that can handle the same media type
        """
        available_models = self.model_manager.get_available_models()
        fallback_models = []
        
        for model_name in available_models:
            # Skip the original model and any with open circuit breakers
            if model_name == original_model:
                continue
            
            if not self.is_model_available(model_name):
                continue
            
            # Check if model handles the same media type
            try:
                model = self.model_manager.get_model(model_name)
                config = model.config
                
                if media_type == 'video' and getattr(config, 'is_video', False):
                    fallback_models.append(model_name)
                elif media_type == 'audio' and getattr(config, 'is_audio', False):
                    fallback_models.append(model_name)
                elif media_type == 'image' and getattr(config, 'is_image', False):
                    fallback_models.append(model_name)
            except Exception as e:
                logger.warning(f"Could not check model {model_name} for fallback: {e}")
                continue
        
        if fallback_models:
            logger.info(f"Found {len(fallback_models)} fallback models for {media_type}: {fallback_models}")
        else:
            logger.warning(f"No fallback models available for {media_type}")
        
        return fallback_models
    
    async def analyze_with_fallback(
        self,
        media_path: str,
        model_name: str,
        media_type: str,
        **kwargs
    ) -> Tuple[Any, str, Optional[str]]:
        """
        Attempt analysis with automatic fallback to alternative models.
        
        Args:
            media_path: Path to the media file
            model_name: Primary model to try
            media_type: Type of media
            **kwargs: Additional arguments to pass to analyze()
            
        Returns:
            Tuple of (result, model_used, warning_message)
            
        Raises:
            InferenceError: If all models fail
        """
        import asyncio
        
        models_to_try = [model_name] + self.get_fallback_models(model_name, media_type)
        
        last_error = None
        for attempt_num, current_model in enumerate(models_to_try, 1):
            if not self.is_model_available(current_model):
                logger.info(f"Skipping unavailable model {current_model} (circuit breaker open)")
                continue
            
            try:
                logger.info(f"Attempt {attempt_num}/{len(models_to_try)}: Trying model {current_model}")
                
                model = self.model_manager.get_model(current_model)
                result = await asyncio.to_thread(model.analyze, media_path, **kwargs)
                
                self.record_success(current_model)
                
                warning_message = None
                if current_model != model_name:
                    warning_message = (
                        f"Primary model '{model_name}' failed. "
                        f"Result provided by fallback model '{current_model}'."
                    )
                    logger.warning(warning_message)
                
                return result, current_model, warning_message
                
            except Exception as e:
                last_error = e
                self.record_failure(current_model, e, media_path)
                logger.error(f"Model {current_model} failed: {e}")
                
                # Don't try more models if it's a media processing error
                # (the media itself is likely corrupted)
                if isinstance(e, MediaProcessingError):
                    logger.error("Media processing error detected. Stopping fallback attempts.")
                    break
                
                continue
        
        # All models failed
        error_msg = (
            f"All {len(models_to_try)} models failed for {media_type} analysis. "
            f"Last error: {last_error}"
        )
        logger.error(error_msg)
        raise InferenceError(error_msg)
    
    def _get_recent_failures(self, model_name: str, within_minutes: int = 5) -> List[ModelFailure]:
        """Get recent failures for a specific model."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=within_minutes)
        return [
            f for f in self._failure_history
            if f.model_name == model_name and f.timestamp > cutoff_time
        ]
    
    def get_failure_summary(self) -> Dict[str, Any]:
        """Get a summary of all failures and circuit breaker status."""
        summary = {
            "total_failures": len(self._failure_history),
            "circuit_breakers_open": list(self._circuit_breaker.keys()),
            "failures_by_model": {},
            "recent_failures": []
        }
        
        # Count failures by model
        for failure in self._failure_history:
            model = failure.model_name
            if model not in summary["failures_by_model"]:
                summary["failures_by_model"][model] = 0
            summary["failures_by_model"][model] += 1
        
        # Get recent failures (last hour)
        recent_cutoff = datetime.utcnow() - timedelta(hours=1)
        summary["recent_failures"] = [
            {
                "model": f.model_name,
                "error_type": f.error_type,
                "error_message": f.error_message,
                "timestamp": f.timestamp.isoformat()
            }
            for f in self._failure_history
            if f.timestamp > recent_cutoff
        ]
        
        return summary
    
    def clear_history(self) -> None:
        """Clear failure history (useful for testing or after manual intervention)."""
        logger.info("Clearing failure history and circuit breakers")
        self._failure_history.clear()
        self._circuit_breaker.clear()
