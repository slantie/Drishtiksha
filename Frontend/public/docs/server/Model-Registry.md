# Model Registry & Manager

## Overview

The **ModelManager** (`src/ml/registry.py`) is the central orchestration component of the Drishtiksha ML inference server. It implements an automated, production-grade model management system that handles discovery, loading, caching, and lifecycle management of all deepfake detection models.

**Purpose:** Eliminate manual model registration, enable dynamic model discovery, support lazy/eager loading strategies, and provide fail-fast validation.

**Design Pattern:** Registry + Factory + Singleton

---

## Core Responsibilities

### 1. Automatic Model Discovery

The ModelManager automatically discovers all detector classes at startup without requiring manual registration.

**Discovery Process:**

```python
def _discover_models(self) -> Dict[str, Type[BaseModel]]:
    """
    Scans src/ml/detectors/ directory to find all BaseModel subclasses.
    
    Benefits:
    - Zero manual registration required
    - New models automatically discovered
    - No stale registry entries
    """
    registry: Dict[str, Type[BaseModel]] = {}
    detectors_package_path = Path(__file__).parent / "detectors"

    # Iterate through all Python modules in detectors/
    for (_, module_name, _) in pkgutil.iter_modules([str(detectors_package_path)]):
        if module_name == "__init__":
            continue
        
        # Dynamically import the module
        module = importlib.import_module(f"src.ml.detectors.{module_name}")
        
        # Find all BaseModel subclasses in the module
        for class_name, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, BaseModel) and cls is not BaseModel:
                logger.debug(f"Discovered model class: {class_name}")
                registry[class_name] = cls
                    
    return registry
```

**Discovery Output:**

```python
{
    "SiglipLSTMV1": <class 'SiglipLSTMV1'>,
    "SiglipLSTMV3": <class 'SiglipLSTMV3'>,
    "SiglipLSTMV4": <class 'SiglipLSTMV4'>,
    "ColorCuesLSTMV1": <class 'ColorCuesLSTMV1'>,
    "EfficientNetB7Detector": <class 'EfficientNetB7Detector'>,
    "EyeblinkDetectorV1": <class 'EyeblinkDetectorV1'>,
    "ScatteringWaveV1": <class 'ScatteringWaveV1'>,
    "MelSpectrogramCNNV2": <class 'MelSpectrogramCNNV2'>,
    "STFTSpectrogramCNNV1": <class 'STFTSpectrogramCNNV1'>,
    "DistilDIREDetectorV1": <class 'DistilDIREDetectorV1'>,
    "MFFMoEDetectorV1": <class 'MFFMoEDetectorV1'>,
    "CrossEfficientViTDetector": <class 'CrossEfficientViTDetector'>,
    "LipFDetectorV1": <class 'LipFDetectorV1'>
}
```

**Key Benefits:**

- **Zero Maintenance**: No manual registry updates needed
- **Extensibility**: Drop new detector file → automatic discovery
- **Safety**: Only valid `BaseModel` subclasses registered
- **Debugging**: Clear logging of discovered classes

### 2. Configuration Mapping

The ModelManager maps user-friendly model names (from `config.yaml`) to discovered detector classes.

**Configuration Structure:**

```yaml
# configs/config.yaml
models:
  SIGLIP-LSTM-V4:                       # User-friendly model name
    class_name: "SiglipLSTMV4"           # Maps to discovered class
    model_path: "models/SigLip-LSTM-v4.pth"
    processor_path: "google/siglip-base-patch16-224"
    # ... other config
```

**Mapping Process:**

```python
# In __init__:
self.model_configs: Dict[str, ModelConfig] = settings.models
# {
#     "SIGLIP-LSTM-V4": SiglipLSTMv4Config(...),
#     "COLOR-CUES-LSTM-V1": ColorCuesConfig(...),
#     "EFFICIENTNET-B7-V1": EfficientNetB7Config(...),
#     ...
# }

# During loading:
model_name = "SIGLIP-LSTM-V4"
model_config = self.model_configs[model_name]  # SiglipLSTMv4Config
class_name = model_config.class_name           # "SiglipLSTMV4"
model_class = self._registry[class_name]       # <class 'SiglipLSTMV4'>
```

### 3. Lazy vs. Eager Loading

The ModelManager supports two loading strategies depending on the use case.

#### Eager Loading (Web Server Startup)

**Purpose:** Preload all models before accepting requests to ensure consistent latency.

```python
def load_models(self):
    """
    Eagerly loads ALL configured models during server startup.
    
    Used by: FastAPI lifespan manager in src/app/main.py
    
    Behavior:
    - Iterates through all model_configs
    - Loads each model into memory (GPU/CPU)
    - Caches loaded instances in self._models
    - Raises ModelRegistryError on ANY failure (fail-fast)
    """
    logger.info("Starting eager loading of all configured models...")
    
    for model_name in self.model_configs.keys():
        try:
            self.load_model(model_name)  # Load and cache
        except Exception as e:
            logger.critical(
                f"❌ FATAL: Failed to load model '{model_name}'. "
                "Server startup will be aborted.",
                exc_info=True
            )
            # Re-raise to prevent server from starting
            raise ModelRegistryError(f"Could not load model '{model_name}'.") from e
    
    logger.info(f"✅ All {len(self._models)} models loaded successfully.")
```

**Server Startup Flow:**

```python
# In src/app/main.py:

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager - handles startup/shutdown."""
    logger.info("🚀 Starting Drishtiksha ML Server...")
    
    try:
        # Eager load ALL models before accepting traffic
        model_manager.load_models()
        
        logger.info("✅ Startup complete. Server ready to accept requests.")
        yield  # Server runs here
        
    except ModelRegistryError as e:
        logger.critical(f"❌ Startup failed: {e}")
        raise  # Exit code 1 → Docker/K8s will restart
    finally:
        logger.info("🔄 Shutting down server...")
```

**Benefits:**

- **Consistent Latency**: First request as fast as any other
- **Fail-Fast**: Missing weights caught at startup, not at first request
- **Health Checks**: External systems can verify all models loaded

#### Lazy Loading (CLI / On-Demand)

**Purpose:** Load models only when needed to save memory and startup time.

```python
def load_model(self, model_name: str) -> BaseModel:
    """
    Lazy loads a single model on-demand.
    
    Used by: CLI tools (src/cli/), API routes with optional model selection
    
    Behavior:
    - Checks if model already cached → return immediately
    - If not cached → load from disk, instantiate, cache, return
    - Next call for same model → instant cache lookup
    """
    # Check cache first (O(1) lookup)
    if model_name in self._models:
        logger.debug(f"Model '{model_name}' already loaded, returning cached.")
        return self._models[model_name]
    
    # Validate model is configured
    if model_name not in self.model_configs:
        available = list(self.model_configs.keys())
        raise ModelRegistryError(
            f"Model '{model_name}' not found in configuration. "
            f"Available models: {available}"
        )
    
    # Load the model
    model_config = self.model_configs[model_name]
    class_name = model_config.class_name
    
    # Lookup class from registry
    model_class = self._registry.get(class_name)
    if not model_class:
        raise ModelRegistryError(
            f"Model class '{class_name}' for model '{model_name}' "
            "not found in registry. Check config.yaml class_name matches Python class."
        )
    
    try:
        logger.info(f"🔄 Lazy loading model '{model_name}' (Class: {class_name})...")
        start_time = time.monotonic()
        
        # CRITICAL: Inject model_name into config
        # This ensures progress events use the consistent model key
        # (e.g., "EFFICIENTNET-B7-V1") instead of class name
        # (e.g., "EfficientNetB7Detector")
        model_config.model_name = model_name
        
        # Instantiate and load weights
        instance = model_class(model_config)
        instance.load()  # Calls detector's load() method
        
        # Cache for future requests
        self._models[model_name] = instance
        
        load_time = time.monotonic() - start_time
        logger.info(f"✅ Successfully loaded '{model_name}' in {load_time:.2f}s.")
        
        return instance
        
    except Exception as e:
        logger.error(f"❌ Failed to load model '{model_name}'.", exc_info=True)
        raise ModelRegistryError(f"Could not load model '{model_name}'.") from e
```

**CLI Usage Example:**

```python
# src/cli/predict.py

def predict(media_path: str, model_name: str):
    """CLI tool for single prediction."""
    # Only load the requested model (lazy loading)
    model = model_manager.get_model(model_name)
    
    result = model.analyze(media_path)
    print(f"Prediction: {result.prediction}")
```

**Benefits:**

- **Memory Efficiency**: Only load what's needed
- **Faster Startup**: CLI tools don't wait for all models
- **Flexible**: User chooses which models to use

### 4. Model Name Injection Fix

**Problem:** Progress events published during inference showed class names (`EfficientNetB7Detector`) instead of user-friendly model names (`EFFICIENTNET-B7-V1`).

**Solution:** Inject `model_name` into config before instantiation.

```python
# In load_model():

model_config = self.model_configs[model_name]  # Get config
model_config.model_name = model_name           # ⚠️ CRITICAL FIX

# Now detector can use config.model_name in progress events:
event_publisher.publish(ProgressEvent(
    media_id=video_id,
    user_id=user_id,
    event="FRAME_ANALYSIS_PROGRESS",
    message=f"Analyzed 50/300 frames",
    data=EventData(
        model_name=self.config.model_name,  # ✅ "EFFICIENTNET-B7-V1"
        progress=50,
        total=300
    )
))
```

**Before Fix:**

```json
{
  "model_name": "EfficientNetB7Detector",  // ❌ Internal class name
  "progress": 50,
  "total": 300
}
```

**After Fix:**

```json
{
  "model_name": "EFFICIENTNET-B7-V1",  // ✅ User-friendly name
  "progress": 50,
  "total": 300
}
```

---

## API Reference

### Initialization

```python
class ModelManager:
    def __init__(self, settings: Settings):
        """
        Initialize the ModelManager with application settings.
        
        Args:
            settings: Pydantic Settings object containing all model configs
        
        Attributes Created:
            _models: Cache of loaded model instances
            model_configs: Dict mapping model names to their configs
            _registry: Dict mapping class names to class objects
        """
```

**Initialization Flow:**

```python
# In src/app/main.py:
settings = Settings()  # Load from .env + config.yaml
model_manager = ModelManager(settings)  # Singleton instance

# Logs:
# ModelManager initialized. Discovered 13 model classes.
# Active models from config: ['SIGLIP-LSTM-V4', 'COLOR-CUES-LSTM-V1', ...]
```

### Core Methods

#### `load_models() -> None`

**Purpose:** Eagerly load all configured models (server startup).

```python
def load_models(self):
    """
    Eagerly loads ALL models specified in configuration.
    
    Usage: Called by FastAPI lifespan manager during startup
    
    Behavior:
    - Iterates through all model_configs
    - Calls load_model() for each
    - Raises ModelRegistryError on ANY failure (fail-fast)
    
    Raises:
        ModelRegistryError: If any model fails to load
    """
```

**Example:**

```python
# Server startup:
model_manager.load_models()

# Logs:
# 🔄 Lazy loading model 'SIGLIP-LSTM-V4' (Class: SiglipLSTMV4)...
# ✅ Loaded Model: 'SIGLIP-LSTM-V4' | Device: 'cuda' | Time: 3.45s.
# ✅ Successfully loaded 'SIGLIP-LSTM-V4' in 3.45s.
# ...
# ✅ All 7 models have been loaded successfully.
```

#### `load_model(model_name: str) -> BaseModel`

**Purpose:** Lazy load a single model on-demand.

```python
def load_model(self, model_name: str) -> BaseModel:
    """
    Lazy loads a single model, caching it for future requests.
    
    Args:
        model_name: Model identifier from config (e.g., "EFFICIENTNET-B7-V1")
    
    Returns:
        Loaded and initialized BaseModel instance
    
    Raises:
        ModelRegistryError: If model config not found or loading fails
    
    Caching:
        - First call: Loads from disk, caches in self._models
        - Subsequent calls: Returns cached instance (instant)
    """
```

**Example:**

```python
# First call (loads from disk):
model = model_manager.load_model("SIGLIP-LSTM-V4")
# Time: ~3.5 seconds

# Second call (cache hit):
model = model_manager.load_model("SIGLIP-LSTM-V4")
# Time: <1 millisecond
```

#### `get_model(name: str) -> BaseModel`

**Purpose:** Retrieve a model, loading it lazily if needed (alias for `load_model`).

```python
def get_model(self, name: str) -> BaseModel:
    """
    Retrieves a model instance, loading lazily if not cached.
    
    Args:
        name: Model identifier from config
    
    Returns:
        BaseModel instance (loaded or cached)
    
    Usage:
        Preferred method for API routes and CLI tools
    """
```

**Example:**

```python
# In API route:
model = model_manager.get_model(model_name)
result = await asyncio.to_thread(model.analyze, media_path)
```

#### `get_available_models() -> List[str]`

**Purpose:** List all configured model names.

```python
def get_available_models(self) -> list[str]:
    """
    Returns list of all model names from configuration.
    
    Returns:
        List of model identifiers (e.g., ["SIGLIP-LSTM-V4", "COLOR-CUES-LSTM-V1"])
    
    Note:
        This shows ALL configured models, not just loaded ones
    """
```

**Example:**

```python
available = model_manager.get_available_models()
# ['SIGLIP-LSTM-V1', 'SIGLIP-LSTM-V3', 'SIGLIP-LSTM-V4', 
#  'COLOR-CUES-LSTM-V1', 'EFFICIENTNET-B7-V1', ...]
```

#### `get_loaded_model_names() -> List[str]`

**Purpose:** List currently loaded model names (memory check).

```python
def get_loaded_model_names(self) -> list[str]:
    """
    Returns list of models currently loaded in memory.
    
    Returns:
        List of model identifiers currently cached
    
    Use Case:
        - Health checks
        - Memory monitoring
        - Debugging
    """
```

**Example:**

```python
# After server startup (eager loading):
loaded = model_manager.get_loaded_model_names()
# ['SIGLIP-LSTM-V4', 'COLOR-CUES-LSTM-V1', 'EFFICIENTNET-B7-V1']

# CLI with lazy loading (only loaded one model):
loaded = model_manager.get_loaded_model_names()
# ['SIGLIP-LSTM-V4']
```

#### `is_model_loaded(name: str) -> bool`

**Purpose:** Check if a model is in cache (without triggering load).

```python
def is_model_loaded(self, name: str) -> bool:
    """
    Check if model is currently loaded in memory.
    
    Args:
        name: Model identifier
    
    Returns:
        True if model is cached, False otherwise
    
    Note:
        Does NOT trigger loading - just checks cache
    """
```

**Example:**

```python
if model_manager.is_model_loaded("SIGLIP-LSTM-V4"):
    print("Model is ready (cached)")
else:
    print("Model will be loaded on first use")
```

#### `get_active_model_configs() -> Dict[str, ModelConfig]`

**Purpose:** Get configuration objects for all models.

```python
def get_active_model_configs(self) -> Dict[str, ModelConfig]:
    """
    Returns configuration objects for all active models.
    
    Returns:
        Dict mapping model names to their Pydantic config objects
    
    Use Case:
        - Introspection
        - API endpoints (/models)
        - Configuration validation
    """
```

**Example:**

```python
configs = model_manager.get_active_model_configs()

for name, config in configs.items():
    print(f"{name}:")
    print(f"  Class: {config.class_name}")
    print(f"  Path: {config.model_path}")
    print(f"  Video: {config.isVideo}")
```

---

## Error Handling

### ModelRegistryError

Custom exception for all registry-related failures.

```python
class ModelRegistryError(Exception):
    """Custom exception for errors related to the model registry."""
    pass
```

**Usage Scenarios:**

1. **Model Not Configured:**

```python
try:
    model = model_manager.get_model("NONEXISTENT-MODEL")
except ModelRegistryError as e:
    # Error: Model 'NONEXISTENT-MODEL' not found in configuration.
    # Available models: ['SIGLIP-LSTM-V4', 'COLOR-CUES-LSTM-V1', ...]
    logger.error(str(e))
```

2. **Class Not Found in Registry:**

```python
# config.yaml has:
# class_name: "NonExistentClass"

try:
    model = model_manager.load_model("BAD-MODEL")
except ModelRegistryError as e:
    # Error: Model class 'NonExistentClass' for model 'BAD-MODEL' not found
    # in registry. Ensure class name in config.yaml matches Python class name.
    logger.error(str(e))
```

3. **Model Loading Failure:**

```python
# Missing model weights file

try:
    model = model_manager.load_model("BROKEN-MODEL")
except ModelRegistryError as e:
    # Error: Could not load model 'BROKEN-MODEL'.
    # Caused by: FileNotFoundError: [Errno 2] No such file or directory: 
    # 'models/missing-weights.pth'
    logger.error(str(e))
```

---

## Integration Examples

### Web Server Integration

```python
# src/app/main.py

from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.config import Settings
from src.ml.registry import ModelManager

# Singleton instances
settings = Settings()
model_manager = ModelManager(settings)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager - handles startup/shutdown."""
    logger.info("🚀 Starting server...")
    
    try:
        # Eager load all models
        model_manager.load_models()
        logger.info("✅ All models loaded. Ready to accept requests.")
        
        yield  # Server runs
        
    except ModelRegistryError as e:
        logger.critical(f"❌ Startup failed: {e}")
        raise  # Exit code 1
    finally:
        logger.info("🔄 Shutting down...")

app = FastAPI(lifespan=lifespan)
```

### API Route Integration

```python
# src/app/routers/analysis.py

from fastapi import APIRouter, Form, UploadFile
from src.ml.registry import model_manager

router = APIRouter()

@router.post("/analyze")
async def analyze_media(
    media: UploadFile,
    model: str = Form(None)
):
    # Use default model if not specified
    model_name = model or settings.DEFAULT_MODEL_NAME
    
    # Get model (lazy load if not cached)
    detector = model_manager.get_model(model_name)
    
    # Offload to thread pool (non-blocking)
    result = await asyncio.to_thread(
        detector.analyze,
        media_path=temp_file.name,
        generate_visualizations=True
    )
    
    return {"success": True, "data": result}
```

### CLI Integration

```python
# src/cli/predict.py

import click
from src.config import Settings
from src.ml.registry import ModelManager

settings = Settings()
model_manager = ModelManager(settings)

@click.command()
@click.argument("media_path")
@click.option("--model", default="SIGLIP-LSTM-V4")
def predict(media_path: str, model: str):
    """Run inference on a single media file."""
    
    # Lazy load only the requested model
    detector = model_manager.get_model(model)
    
    result = detector.analyze(media_path)
    
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Processing Time: {result.processing_time:.2f}s")

if __name__ == "__main__":
    predict()
```

---

## Design Patterns

### 1. Registry Pattern

**Purpose:** Centralized storage of model class references.

**Implementation:**

```python
# Automatic discovery populates registry
self._registry: Dict[str, Type[BaseModel]] = {
    "SiglipLSTMV4": <class 'SiglipLSTMV4'>,
    "EfficientNetB7Detector": <class 'EfficientNetB7Detector'>,
    ...
}

# Lookup during loading
model_class = self._registry[class_name]
instance = model_class(config)
```

**Benefits:**

- **Decoupling**: API layer doesn't need to import every detector class
- **Discoverability**: All available models in one place
- **Validation**: Only valid `BaseModel` subclasses registered

### 2. Factory Pattern

**Purpose:** Centralized instantiation of model objects.

**Implementation:**

```python
def load_model(self, model_name: str) -> BaseModel:
    # Lookup configuration
    config = self.model_configs[model_name]
    
    # Lookup class
    model_class = self._registry[config.class_name]
    
    # Factory: Create instance
    instance = model_class(config)
    instance.load()
    
    return instance
```

**Benefits:**

- **Abstraction**: Caller doesn't need to know constructor details
- **Consistency**: All models instantiated the same way
- **Validation**: Config injection ensures proper initialization

### 3. Singleton Pattern

**Purpose:** Single ModelManager instance shared across application.

**Implementation:**

```python
# In src/app/main.py:
settings = Settings()
model_manager = ModelManager(settings)  # Created once

# Imported everywhere:
from src.app.main import model_manager
```

**Benefits:**

- **Shared Cache**: All routes use same loaded models
- **Memory Efficiency**: No duplicate model instances
- **State Management**: Single source of truth for loaded models

### 4. Lazy Initialization Pattern

**Purpose:** Defer expensive operations (model loading) until needed.

**Implementation:**

```python
def load_model(self, model_name: str) -> BaseModel:
    # Check cache first
    if model_name in self._models:
        return self._models[model_name]  # ⚡ Instant
    
    # Load on first access
    instance = self._load_from_disk(model_name)  # 🐢 Slow
    self._models[model_name] = instance          # Cache
    return instance
```

**Benefits:**

- **Faster Startup**: CLI doesn't wait for all models
- **Memory Efficiency**: Only load what's used
- **Flexibility**: User controls which models to load

---

## Performance Considerations

### Memory Footprint

**Typical Model Sizes:**

| Model | Parameters | FP32 Size | GPU Memory (Peak) |
|-------|------------|-----------|-------------------|
| SIGLIP-LSTM-V4 | ~150M | ~600 MB | ~1.2 GB |
| ColorCues-LSTM-V1 | ~5M | ~20 MB | ~100 MB |
| EfficientNet-B7-V1 | ~66M | ~264 MB | ~800 MB |
| MFF-MoE-V1 | ~195M | ~780 MB | ~3.5 GB |

**Total (All 11 Models Loaded):**

- **Model Weights**: ~4-5 GB
- **Peak GPU Memory**: ~12-15 GB (with batch processing)

**Optimization:**

```python
# Eager loading (server): Load all upfront
model_manager.load_models()  # ~12 GB GPU RAM

# Lazy loading (CLI): Load only what's needed
model = model_manager.get_model("SIGLIP-LSTM-V4")  # ~1.2 GB GPU RAM
```

### Loading Time

**Eager Loading Benchmark (7 models):**

```text
🔄 Lazy loading model 'SIGLIP-LSTM-V4'...
✅ Successfully loaded in 3.42s.

🔄 Lazy loading model 'COLOR-CUES-LSTM-V1'...
✅ Successfully loaded in 1.15s.

🔄 Lazy loading model 'EFFICIENTNET-B7-V1'...
✅ Successfully loaded in 2.87s.

... (4 more models)

✅ All 7 models loaded in 18.5s total.
```

**Lazy Loading Benchmark (single model):**

```text
# First call (cold):
model = model_manager.get_model("SIGLIP-LSTM-V4")
# Time: 3.42s

# Second call (cached):
model = model_manager.get_model("SIGLIP-LSTM-V4")
# Time: 0.0001s (instant dictionary lookup)
```

---

## Summary

The ModelManager is a sophisticated, production-ready component that automates the entire model lifecycle:

✅ **Automatic Discovery** - No manual registration required  
✅ **Lazy/Eager Loading** - Flexible strategies for different use cases  
✅ **Fail-Fast Validation** - Catches errors at startup, not runtime  
✅ **Caching** - Instant access to already-loaded models  
✅ **Type-Safe** - Pydantic validation ensures correct configuration  
✅ **Extensible** - Drop new detector → automatic integration  
✅ **Observable** - Rich logging and introspection APIs  

**Key Takeaway:** Add a new deepfake detection model by creating a single Python file in `src/ml/detectors/` and adding its configuration to `config.yaml`. The ModelManager handles the rest automatically.
