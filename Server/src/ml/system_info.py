# src/ml/system_info.py

import os
import time
import platform
import sys
import logging
from typing import Dict, Any, List
from functools import wraps

# These imports are conditional to prevent crashes if they aren't installed.
try:
    import psutil
except ImportError:
    psutil = None
try:
    import torch
except ImportError:
    torch = None
try:
    import cpuinfo
except ImportError:
    cpuinfo = None

from src.app.schemas import DeviceInfo, SystemInfo, ModelInfo
from src.config import settings
from src.ml.registry import ModelManager

logger = logging.getLogger(__name__)


def _timed_cache(ttl_seconds: int):
    """
    A lightweight decorator for time-based caching.
    Caches the result of a function for a specified number of seconds.
    """
    def decorator(func):
        cache = {}
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.monotonic()
            if "result" not in cache or (now - cache.get("timestamp", 0)) > ttl_seconds:
                cache["result"] = func(*args, **kwargs)
                cache["timestamp"] = now
            return cache["result"]
        return wrapper
    return decorator


class SystemMonitor:
    """
    REFACTORED class-based system monitor.

    Provides cached, efficient access to system, device, and model statistics.
    Manages application state like start time internally.
    """
    def __init__(self):
        self._start_time = time.monotonic()
        logger.info("SystemMonitor initialized and application start time recorded.")

    @property
    def uptime(self) -> float:
        """Calculates the server uptime in seconds."""
        return round(time.monotonic() - self._start_time, 2)

    @_timed_cache(ttl_seconds=10)
    def get_device_info(self) -> DeviceInfo:
        """Get detailed information about the compute device. Cached for 10 seconds."""
        if not torch:
            return DeviceInfo(type="cpu", name="PyTorch not available")

        device_type = settings.device.lower()
        if device_type == "cuda" and torch.cuda.is_available():
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            total_mem = props.total_memory / (1024**3)
            reserved_mem = torch.cuda.memory_reserved(device) / (1024**3)
            return DeviceInfo(
                type="cuda",
                name=torch.cuda.get_device_name(device),
                total_memory=round(total_mem, 2),
                used_memory=round(reserved_mem, 2),
                free_memory=round(total_mem - reserved_mem, 2),
                memory_usage_percent=round((reserved_mem / total_mem) * 100, 2),
                compute_capability=f"{props.major}.{props.minor}",
                cuda_version=torch.version.cuda
            )
        else:
            fallback = ""
            if device_type == "cuda" and not torch.cuda.is_available():
                fallback = " (Fallback: CUDA not available)"
            return DeviceInfo(type="cpu", name=f"{self._get_cpu_name()}{fallback}")

    def _get_cpu_name(self) -> str:
        """Get a detailed CPU name using the cpuinfo library as the primary source."""
        if cpuinfo:
            try:
                info = cpuinfo.get_cpu_info()
                if 'brand_raw' in info:
                    return info['brand_raw']
            except Exception as e:
                logger.warning(f"Could not get CPU name from cpuinfo: {e}")
        return platform.processor() or "Unknown CPU"

    @_timed_cache(ttl_seconds=5)
    def get_system_info(self) -> SystemInfo:
        """Get system resource information. Cached for 5 seconds."""
        if not psutil:
            return SystemInfo(
                python_version=sys.version.split()[0], platform=platform.platform(),
                cpu_count=os.cpu_count() or 1, total_ram=0.0, used_ram=0.0, ram_usage_percent=0.0,
                uptime_seconds=self.uptime
            )
        
        memory = psutil.virtual_memory()
        return SystemInfo(
            python_version=sys.version.split()[0], platform=platform.platform(),
            cpu_count=psutil.cpu_count(),
            total_ram=round(memory.total / (1024**3), 2),
            used_ram=round(memory.used / (1024**3), 2),
            ram_usage_percent=round(memory.percent, 2),
            uptime_seconds=self.uptime
        )

    def get_models_info(self, manager: ModelManager) -> List[ModelInfo]:
        """Get detailed information about all active models."""
        model_infos = []
        loaded_models = manager.get_loaded_model_names()
        active_configs = manager.get_active_model_configs()
        
        for name, config in active_configs.items():
            model_infos.append(ModelInfo(
                name=name, class_name=config.class_name, description=config.description,
                loaded=(name in loaded_models), device=config.device, model_path=str(config.model_path),
                isAudio=config.isAudio, isVideo=config.isVideo, isImage=config.isImage
            ))
        return model_infos

    def get_server_configuration(self) -> Dict[str, Any]:
        """Get server configuration summary."""
        config = {
            "project_name": settings.project_name,
            "device": settings.device,
            "default_model": settings.default_model_name,
            "active_models": settings.active_model_list,
        }
        if torch:
            config["torch_version"] = torch.__version__
            config["cuda_available"] = torch.cuda.is_available()
        return config

# --- Singleton Instance ---
# The application will import and use this single instance.
system_monitor = SystemMonitor()