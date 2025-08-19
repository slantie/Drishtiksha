# src/ml/system_info.py

import time
import platform
import sys
from typing import Dict, Any, Optional

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None

from src.app.schemas import DeviceInfo, SystemInfo, ModelInfo
from src.config import settings

# Track server start time
_start_time = time.time()

def get_device_info() -> DeviceInfo:
    """Get detailed information about the compute device."""
    if torch is None:
        return DeviceInfo(
            type="cpu",
            name="PyTorch not available",
            total_memory=None,
            used_memory=None,
            free_memory=None,
            memory_usage_percent=None,
            compute_capability=None,
            cuda_version=None
        )
    
    device_type = settings.device.lower()
    
    # Respect user's device preference from environment variable
    if device_type == "cpu":
        # User explicitly wants CPU processing - get better CPU name
        cpu_name = _get_cpu_name()
        return DeviceInfo(
            type="cpu",
            name=cpu_name,
            total_memory=None,
            used_memory=None,
            free_memory=None,
            memory_usage_percent=None,
            compute_capability=None,
            cuda_version=None
        )
    elif device_type == "cuda" and torch.cuda.is_available():
        # User wants CUDA and it's available
        device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)
        
        # Memory information
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB
        allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)  # GB
        reserved_memory = torch.cuda.memory_reserved(device) / (1024**3)  # GB
        free_memory = total_memory - reserved_memory
        memory_usage_percent = (reserved_memory / total_memory) * 100
        
        # Compute capability
        props = torch.cuda.get_device_properties(device)
        compute_capability = f"{props.major}.{props.minor}"
        
        # CUDA version
        cuda_version = torch.version.cuda
        
        return DeviceInfo(
            type="cuda",
            name=device_name,
            total_memory=round(total_memory, 2),
            used_memory=round(reserved_memory, 2),
            free_memory=round(free_memory, 2),
            memory_usage_percent=round(memory_usage_percent, 2),
            compute_capability=compute_capability,
            cuda_version=cuda_version
        )
    else:
        # Fallback to CPU (either user wanted CUDA but it's not available, or invalid device type)
        fallback_reason = "CUDA not available" if device_type == "cuda" else f"Invalid device type: {device_type}"
        cpu_name = _get_cpu_name()
        return DeviceInfo(
            type="cpu",
            name=f"{cpu_name} (Fallback: {fallback_reason})",
            total_memory=None,
            used_memory=None,
            free_memory=None,
            memory_usage_percent=None,
            compute_capability=None,
            cuda_version=None
        )

def _get_cpu_name() -> str:
    """Get a better CPU name, especially on Windows."""
    try:
        # Try to get CPU name from Windows registry first
        if platform.system() == "Windows":
            import winreg
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                  r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                    cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
                    return cpu_name.strip()
            except Exception as e:
                print(f"Registry method failed: {e}")
            
            # Try WMI as alternative for Windows
            try:
                import subprocess
                result = subprocess.run([
                    "wmic", "cpu", "get", "name", "/value"
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if line.startswith('Name='):
                            cpu_name = line.split('=', 1)[1].strip()
                            if cpu_name:
                                print(f"WMI method found: {cpu_name}")
                                return cpu_name
            except Exception as e:
                print(f"WMI method failed: {e}")
        
        # Fallback to cpuinfo if available
        if psutil is not None:
            try:
                import cpuinfo
                info = cpuinfo.get_cpu_info()
                if 'brand_raw' in info:
                    return info['brand_raw']
                elif 'brand' in info:
                    return info['brand']
            except ImportError:
                pass
        
        # Final fallback to platform.processor()
        processor = platform.processor()
        if processor:
            print(f"Platform fallback: {processor}")
            return processor
        
        return "Unknown CPU"
        
    except Exception as e:
        print(f"Overall exception in _get_cpu_name: {e}")
        return "Unknown CPU"

def get_system_info() -> SystemInfo:
    """Get system resource information."""
    if psutil is None:
        return SystemInfo(
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            cpu_count=1,
            total_ram=0.0,
            used_ram=0.0,
            ram_usage_percent=0.0,
            uptime_seconds=round(time.time() - _start_time, 2)
        )
    
    # Memory information
    memory = psutil.virtual_memory()
    total_ram = memory.total / (1024**3)  # GB
    used_ram = memory.used / (1024**3)    # GB
    ram_usage_percent = memory.percent
    
    # Uptime
    uptime_seconds = time.time() - _start_time
    
    return SystemInfo(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        cpu_count=psutil.cpu_count(),
        total_ram=round(total_ram, 2),
        used_ram=round(used_ram, 2),
        ram_usage_percent=round(ram_usage_percent, 2),
        uptime_seconds=round(uptime_seconds, 2)
    )

def get_model_info(manager) -> list[ModelInfo]:
    """Get detailed information about all models."""
    model_infos = []
    
    for name, config in manager.model_configs.items():
        is_loaded = name in manager._models
        
        # Try to get memory usage if model is loaded
        memory_usage_mb = None
        if is_loaded and hasattr(manager._models[name], 'model'):
            try:
                model = manager._models[name].model
                if next(model.parameters()).is_cuda:
                    # Rough estimation of model memory usage
                    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
                    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
                    memory_usage_mb = (param_size + buffer_size) / (1024**2)
            except Exception:
                pass  # Memory estimation failed, leave as None
        
        model_info = ModelInfo(
            name=name,
            class_name=config.class_name,
            description=config.description,
            loaded=is_loaded,
            device=config.device,
            model_path=config.model_path,
            isDetailed=config.isDetailed,  # Include detailed analysis capability
            memory_usage_mb=round(memory_usage_mb, 2) if memory_usage_mb else None,
            load_time=None,  # TODO: Track this in model loading
            inference_count=0  # TODO: Track this in model inference
        )
        model_infos.append(model_info)
    
    return model_infos

def get_server_configuration() -> Dict[str, Any]:
    """Get server configuration summary."""
    config = {
        "project_name": settings.project_name,
        "device": settings.device,
        "default_model": settings.default_model_name,
        "active_models": settings.active_model_list,
        "total_configured_models": len(settings.models),
        "active_models_count": len(settings.active_model_list),
        "python_version": sys.version.split()[0],
    }
    
    if torch is not None:
        config.update({
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        })
    else:
        config.update({
            "torch_version": "Not available",
            "cuda_available": False,
            "cuda_device_count": 0
        })
    
    return config

def reset_start_time():
    """Reset the server start time (for testing purposes)."""
    global _start_time
    _start_time = time.time()
