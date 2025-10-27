# BaseModel Contract & Detector Pattern

## Overview

**BaseModel** (`src/ml/base.py`) is an abstract base class that defines the contract for all deepfake detection models in Drishtiksha. It enforces a consistent interface across video, audio, image, and multimodal detectors using the **Strategy Pattern**.

**Purpose:** Ensure all detectors implement standardized `load()` and `analyze()` methods, enabling polymorphic model usage throughout the system.

**Design Pattern:** Strategy Pattern + Template Method

---

## Core Concepts

### Why an Abstract Base Class?

**Without BaseModel:**

```python
# ❌ Inconsistent interfaces
class Model1:
    def initialize(self):  # Different method name
        pass
    
    def detect(self, path):  # Different method name
        return {"result": "fake"}  # Different return type

class Model2:
    def setup(self):  # Different method name
        pass
    
    def predict(self, file):  # Different method name
        return "FAKE"  # Different return type
```

**With BaseModel:**

```python
# ✅ Consistent interfaces
class Model1(BaseModel):
    def load(self):  # Standard method name
        pass
    
    def analyze(self, media_path, **kwargs) -> AnalysisResult:
        return VideoAnalysisResult(...)  # Type-safe return

class Model2(BaseModel):
    def load(self):  # Standard method name
        pass
    
    def analyze(self, media_path, **kwargs) -> AnalysisResult:
        return AudioAnalysisResult(...)  # Type-safe return
```

**Benefits:**

- **Polymorphism**: Treat all models uniformly (code to interface, not implementation)
- **Type Safety**: Enforce return types via `AnalysisResult` union
- **Consistency**: Guarantee all models have `load()` and `analyze()`
- **Maintainability**: Add new models without changing existing code

### Strategy Pattern

The Strategy Pattern allows selecting different detection algorithms at runtime:

```python
# Same interface, different implementations
models = {
    "SIGLIP-LSTM-V4": SiglipLSTMV4(...),
    "MEL-Spectrogram-CNN": MelSpectrogramCNNV1(...),
    "DistilDIRE": DistilDIREDetectorV1(...)
}

# Polymorphic usage
selected_model = models[user_choice]
result = selected_model.analyze(media_path)  # Same call signature

# Works for ANY model that implements BaseModel
```

**Architecture:**

```text
┌─────────────────────────────────────────────────────────┐
│                    BaseModel (ABC)                      │
│  - Abstract: load()                                     │
│  - Abstract: analyze(media_path) -> AnalysisResult      │
│  - Concrete: get_info() -> Dict                         │
└──────────────────┬──────────────────────────────────────┘
                   │
     ┌─────────────┼─────────────┬──────────────┐
     │             │             │              │
┌────▼────┐  ┌────▼────┐  ┌────▼────┐  ┌──────▼──────┐
│ Video   │  │ Audio   │  │ Image   │  │ Multimodal  │
│Detectors│  │Detectors│  │Detectors│  │ Detectors   │
└────┬────┘  └────┬────┘  └────┬────┘  └──────┬──────┘
     │             │             │              │
     │             │             │              │
 ┌───▼───┐    ┌───▼───┐    ┌───▼───┐     ┌───▼───┐
 │Siglip │    │  MEL  │    │DistDIRE│    │MFF-MoE│
 │ LSTM  │    │ Spec  │    │   V1   │    │  V1   │
 └───────┘    └───────┘    └────────┘    └───────┘
```

---

## BaseModel Class

### Class Definition

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Union

from src.config import ModelConfig
from src.app.schemas import VideoAnalysisResult, AudioAnalysisResult

# Type hint for standardized analysis output
AnalysisResult = Union[VideoAnalysisResult, AudioAnalysisResult]


class BaseModel(ABC):
    """
    Abstract Base Class for all machine learning models.

    Defines a clean, simple, and powerful contract that all model
    implementations must adhere to. Designed to be media-type agnostic,
    supporting video, audio, images, and more.
    """
```

**Key Features:**

- **ABC Inheritance**: Enforces abstract method implementation
- **Type Hints**: Clear contracts with `AnalysisResult` union type
- **Media Agnostic**: Works with video, audio, image, multimodal data

### Constructor

```python
def __init__(self, config: ModelConfig):
    """
    Initializes the model with its specific, type-validated configuration.

    Args:
        config (ModelConfig): A Pydantic model containing the configuration
                              for this specific model instance.
    
    Attributes:
        self.config: Validated Pydantic configuration object
        self.model: Placeholder for the ML model (set in load())
        self.processor: Placeholder for preprocessing components (set in load())
        self.device: Target device ("cuda" or "cpu")
    """
    self.config = config
    self.model: Any = None        # Set in load()
    self.processor: Any = None    # Set in load() if needed
    self.device = config.device   # "cuda" or "cpu"
```

**Configuration Validation:**

```python
# config is a discriminated union (Pydantic validates at runtime)
config = ModelConfig  # One of: SiglipLSTMv4Config, MelSpectrogramCNNConfig, etc.

# Pydantic ensures:
# - All required fields present (model_path, class_name, etc.)
# - Correct types (paths exist, integers are valid, etc.)
# - Model-specific fields validated (e.g., num_frames for video models)

detector = SiglipLSTMV4(config)  # ✅ Type-safe initialization
```

**Why `Any` for `model` and `processor`?**

```python
# Different models use different frameworks:
self.model: Any = None

# Could be:
# - torch.nn.Module (PyTorch models)
# - timm.models.VisionTransformer (timm models)
# - torch.jit.ScriptModule (TorchScript)
# - transformers.AutoModel (Hugging Face)

# Python's type system can't express this diversity easily
# Using Any provides flexibility while maintaining runtime type safety
```

---

## Abstract Methods

### 1. load()

**Contract:**

```python
@abstractmethod
def load(self) -> None:
    """
    Loads the model, weights, and any necessary processors into memory.
    This method is called once at server startup for each active model.
    It should set self.model and any other required components.
    
    Raises:
        RuntimeError: If model files are missing or corrupted
        torch.cuda.CUDAError: If GPU memory insufficient
    """
    pass
```

**Responsibilities:**

1. **Load Model Weights**: Load checkpoint/weights from `config.model_path`
2. **Initialize Processors**: Setup preprocessing components (transforms, tokenizers, etc.)
3. **Move to Device**: Transfer model to GPU/CPU (`model.to(self.device)`)
4. **Set Eval Mode**: Switch to inference mode (`model.eval()`)
5. **Validate Loading**: Ensure all components loaded successfully

**Example Implementation (Video Detector):**

```python
class SiglipLSTMV4(BaseModel):
    def load(self) -> None:
        logger.info(f"Loading SIGLIP-LSTM-V4 model from {self.config.model_path}")
        
        # 1. Load processor (SigLIP image processor)
        self.processor = AutoProcessor.from_pretrained(
            self.config.processor_path
        )
        logger.info("✅ SigLIP processor loaded successfully")
        
        # 2. Initialize model architecture
        model = SiglipLSTMModel(
            base_model_path=self.config.model_definition.base_model_path,
            lstm_hidden_size=self.config.model_definition.lstm_hidden_size,
            lstm_num_layers=self.config.model_definition.lstm_num_layers,
            num_classes=self.config.model_definition.num_classes,
            dropout_rate=self.config.model_definition.dropout_rate
        )
        
        # 3. Load trained weights
        checkpoint = torch.load(
            self.config.model_path,
            map_location=self.device,
            weights_only=True
        )
        model.load_state_dict(checkpoint)
        logger.info("✅ Model weights loaded successfully")
        
        # 4. Move to device and set eval mode
        self.model = model.to(self.device)
        self.model.eval()
        
        logger.info(f"🚀 SIGLIP-LSTM-V4 ready on {self.device}")
```

**Example Implementation (Audio Detector):**

```python
class MelSpectrogramCNNV1(BaseModel):
    def load(self) -> None:
        logger.info(f"Loading MEL-Spectrogram-CNN model from {self.config.model_path}")
        
        # 1. Initialize ResNet34 architecture
        from torchvision.models import resnet34
        model = resnet34(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Binary classification
        
        # 2. Load trained weights
        checkpoint = torch.load(
            self.config.model_path,
            map_location=self.device,
            weights_only=True
        )
        model.load_state_dict(checkpoint)
        
        # 3. Move to device and set eval mode
        self.model = model.to(self.device)
        self.model.eval()
        
        # Audio models don't need separate preprocessors
        # (preprocessing done in analyze() with librosa)
        self.processor = None
        
        logger.info(f"🚀 MEL-Spectrogram-CNN ready on {self.device}")
```

**Loading Strategies:**

```python
# Lazy Loading (default):
# - load() called only when model first used
# - Saves memory for inactive models
model = registry.get_model("SIGLIP-LSTM-V4")  # Triggers load() if not loaded

# Eager Loading:
# - load() called at server startup
# - All active models loaded immediately
registry.load_models()  # Loads all ACTIVE_MODELS at once
```

### 2. analyze()

**Contract:**

```python
@abstractmethod
def analyze(
    self, 
    media_path: str, 
    generate_visualizations: bool = False, 
    **kwargs
) -> AnalysisResult:
    """
    Performs a comprehensive analysis on the given media file.

    This is the single, unified entry point for all model inference. The method
    should perform all necessary processing and return a structured Pydantic
    model containing the complete results.

    Args:
        media_path (str): 
            The local file path to the media to be analyzed.
            
        generate_visualizations (bool): 
            If True, generate visualization videos/images.
            Defaults to False for API calls (raw data only).
            Set to True for CLI usage.
            
        **kwargs: 
            Catches additional parameters that might be sent from the API,
            such as:
            - video_id (str): Unique media identifier for event publishing
            - user_id (str): User identifier for event publishing
            - threshold (float): Custom detection threshold

    Returns:
        AnalysisResult: One of:
            - VideoAnalysisResult (for video/image models)
            - AudioAnalysisResult (for audio models)

    Raises:
        FileNotFoundError: If media_path does not exist
        ValueError: If media format unsupported
        RuntimeError: If inference fails
    """
    pass
```

#### Return Type: AnalysisResult Union

```python
# AnalysisResult is a discriminated union:
AnalysisResult = Union[VideoAnalysisResult, AudioAnalysisResult]

# Pydantic schemas ensure type safety:
class VideoAnalysisResult(BaseAnalysisResult):
    prediction: str                    # "REAL" or "FAKE"
    confidence: float                  # 0.0 to 1.0
    processing_time: float             # seconds
    media_type: str = "video"
    frame_count: Optional[int]
    frames_analyzed: int
    frame_predictions: List[FramePrediction]
    metrics: Union[SequenceBasedMetrics, FrameBasedMetrics, ...]
    visualization_path: Optional[str]

class AudioAnalysisResult(BaseAnalysisResult):
    prediction: str                    # "REAL" or "FAKE"
    confidence: float                  # 0.0 to 1.0
    processing_time: float             # seconds
    media_type: str = "audio"
    properties: AudioProperties
    pitch: PitchMetrics
    energy: EnergyMetrics
    spectral: SpectralMetrics
    voice_quality: Optional[VoiceQualityMetrics]
    visualization: Optional[AudioVisualization]
```

**Example Implementation (Video Detector):**

```python
class SiglipLSTMV4(BaseModel):
    def analyze(
        self, 
        media_path: str, 
        generate_visualizations: bool = False, 
        **kwargs
    ) -> VideoAnalysisResult:
        """
        Analyze video using rolling window approach with SigLIP features.
        """
        start_time = time.time()
        
        # Extract kwargs for event publishing
        video_id = kwargs.get("video_id")
        user_id = kwargs.get("user_id")
        
        # 1. Extract frames
        frames = self._extract_frames(media_path, self.config.num_frames)
        logger.info(f"Extracted {len(frames)} frames from video")
        
        # 2. Process frames with rolling window
        window_size = self.config.rolling_window_size
        num_windows = len(frames) - window_size + 1
        window_predictions = []
        
        for i in range(num_windows):
            # Get window of frames
            window = frames[i:i + window_size]
            
            # Preprocess with SigLIP processor
            inputs = self.processor(
                images=window,
                return_tensors="pt"
            ).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs, dim=1)
                fake_prob = probs[0][1].item()
            
            # Store prediction
            window_predictions.append(FramePrediction(
                index=i,
                score=fake_prob,
                prediction="FAKE" if fake_prob > 0.5 else "REAL"
            ))
            
            # Publish progress (every 10 windows)
            if (i + 1) % 10 == 0 and video_id and user_id:
                event_publisher.publish(ProgressEvent(
                    media_id=video_id,
                    user_id=user_id,
                    event="FRAME_ANALYSIS_PROGRESS",
                    message=f"Analyzed window {i + 1}/{num_windows}",
                    data=EventData(
                        model_name=self.config.model_name,
                        progress=i + 1,
                        total=num_windows
                    )
                ))
        
        # 3. Aggregate results
        avg_score = np.mean([p.score for p in window_predictions])
        final_prediction = "FAKE" if avg_score > 0.5 else "REAL"
        
        # 4. Calculate metrics
        metrics = SequenceBasedMetrics(
            mean_score=float(avg_score),
            std_score=float(np.std([p.score for p in window_predictions])),
            max_score=float(max(p.score for p in window_predictions)),
            min_score=float(min(p.score for p in window_predictions)),
            fake_ratio=float(sum(1 for p in window_predictions if p.prediction == "FAKE") / len(window_predictions))
        )
        
        # 5. Generate visualization (optional)
        viz_path = None
        if generate_visualizations:
            viz_path = self._create_visualization(
                media_path, 
                window_predictions, 
                output_dir=settings.storage_path
            )
        
        processing_time = time.time() - start_time
        
        # 6. Return structured result
        return VideoAnalysisResult(
            prediction=final_prediction,
            confidence=avg_score if final_prediction == "FAKE" else (1.0 - avg_score),
            processing_time=processing_time,
            frame_count=len(frames),
            frames_analyzed=num_windows,
            frame_predictions=window_predictions,
            metrics=metrics,
            visualization_path=viz_path
        )
```

**Example Implementation (Audio Detector):**

```python
class MelSpectrogramCNNV1(BaseModel):
    def analyze(
        self, 
        media_path: str, 
        generate_visualizations: bool = False, 
        **kwargs
    ) -> AudioAnalysisResult:
        """
        Analyze audio by generating mel-spectrograms and classifying chunks.
        """
        start_time = time.time()
        
        video_id = kwargs.get("video_id")
        user_id = kwargs.get("user_id")
        
        # 1. Extract audio from video
        if video_id and user_id:
            event_publisher.publish(ProgressEvent(
                media_id=video_id,
                user_id=user_id,
                event="AUDIO_EXTRACTION_START",
                message="Extracting audio track",
                data=EventData(model_name=self.config.model_name)
            ))
        
        audio, sr = librosa.load(media_path, sr=self.config.sampling_rate)
        duration = len(audio) / sr
        
        if video_id and user_id:
            event_publisher.publish(ProgressEvent(
                media_id=video_id,
                user_id=user_id,
                event="AUDIO_EXTRACTION_COMPLETE",
                message=f"Audio extracted ({duration:.1f}s)",
                data=EventData(
                    model_name=self.config.model_name,
                    details={"duration_seconds": duration}
                )
            ))
        
        # 2. Generate mel-spectrogram chunks
        chunk_length = int(self.config.chunk_duration_s * sr)
        num_chunks = len(audio) // chunk_length
        chunk_predictions = []
        
        for i in range(num_chunks):
            # Extract chunk
            chunk = audio[i * chunk_length:(i + 1) * chunk_length]
            
            # Generate mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=chunk,
                sr=sr,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                n_mels=self.config.n_mels
            )
            
            # Convert to image tensor
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            img_tensor = self._spectrogram_to_tensor(mel_db)
            img_tensor = img_tensor.to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                fake_prob = probs[0][1].item()
            
            chunk_predictions.append(fake_prob)
        
        # 3. Aggregate results
        avg_score = np.mean(chunk_predictions)
        final_prediction = "FAKE" if avg_score > 0.5 else "REAL"
        
        # 4. Calculate audio properties and metrics
        properties = AudioProperties(
            duration_seconds=duration,
            sample_rate=sr,
            channels=1
        )
        
        pitch_metrics = self._calculate_pitch_metrics(audio, sr)
        energy_metrics = self._calculate_energy_metrics(audio)
        spectral_metrics = self._calculate_spectral_metrics(audio, sr)
        
        processing_time = time.time() - start_time
        
        # 5. Return structured result
        return AudioAnalysisResult(
            prediction=final_prediction,
            confidence=avg_score if final_prediction == "FAKE" else (1.0 - avg_score),
            processing_time=processing_time,
            properties=properties,
            pitch=pitch_metrics,
            energy=energy_metrics,
            spectral=spectral_metrics,
            voice_quality=None,  # Optional advanced metrics
            visualization=None   # Could add spectrogram visualization
        )
```

**kwargs Usage Patterns:**

```python
# API Call (with event publishing):
result = detector.analyze(
    media_path="video.mp4",
    generate_visualizations=False,  # Don't generate viz for API
    video_id="abc123",              # For event publishing
    user_id="user456"               # For event publishing
)

# CLI Call (with visualization):
result = detector.analyze(
    media_path="video.mp4",
    generate_visualizations=True,   # Generate viz for user
    # No video_id/user_id (no event publishing)
)

# Conditional publishing inside analyze():
if video_id and user_id:
    event_publisher.publish(...)  # Only publish if IDs provided
```

---

## Concrete Method

### get_info()

**Implementation:**

```python
def get_info(self) -> Dict[str, Any]:
    """
    Returns metadata about the model, derived from its config.
    
    Returns:
        Dict containing:
        - model_name: User-friendly model name
        - class_name: Python class name
        - model_path: Path to weights file
        - device: Current device (cuda/cpu)
    """
    return {
        "model_name": self.config.description,
        "class_name": self.config.class_name,
        "model_path": self.config.model_path,
        "device": self.device,
    }
```

**Usage:**

```python
# Get model metadata
info = detector.get_info()
print(info)
# {
#   "model_name": "SIGLIP-LSTM-V4",
#   "class_name": "SiglipLSTMV4",
#   "model_path": "/app/models/SigLip-LSTM-v4.pth",
#   "device": "cuda"
# }

# Used in /models API endpoint
@router.get("/models")
def list_models():
    return [
        {
            **model.get_info(),
            "loaded": registry.is_model_loaded(name)
        }
        for name, model in registry._models.items()
    ]
```

---

## Model Configuration

### Discriminated Union

**Configuration uses Pydantic discriminated unions for type safety:**

```python
# ModelConfig is a union of all model-specific configs
ModelConfig = Annotated[
    Union[
        ColorCuesConfig,
        SiglipLSTMv1Config,
        SiglipLSTMv3Config,
        SiglipLSTMv4Config,
        EfficientNetB7Config,
        # ... 11 total configs
    ],
    Field(discriminator="class_name"),  # Discriminator field
]
```

**How Discrimination Works:**

```yaml
# config.yaml
models:
  SIGLIP-LSTM-V4:
    class_name: "SiglipLSTMV4"  # ← Discriminator determines config type
    description: "SIGLIP-LSTM-V4"
    model_path: "models/SigLip-LSTM-v4.pth"
    processor_path: "google/siglip-so400m-patch14-384"
    num_frames: 300
    rolling_window_size: 8
    model_definition:
      base_model_path: "google/siglip-so400m-patch14-384"
      lstm_hidden_size: 256
      lstm_num_layers: 2
      num_classes: 2
      dropout_rate: 0.5
```

**Pydantic automatically selects correct config type:**

```python
# Parsing config.yaml
config_dict = yaml.safe_load(open("config.yaml"))
model_config_dict = config_dict["models"]["SIGLIP-LSTM-V4"]

# Pydantic sees class_name="SiglipLSTMV4"
# → Parses as SiglipLSTMv4Config (with all its specific fields)
config: ModelConfig = parse_obj(model_config_dict)

# Type is now SiglipLSTMv4Config, not generic ModelConfig
assert isinstance(config, SiglipLSTMv4Config)

# Access model-specific fields
print(config.processor_path)        # ✅ Works
print(config.rolling_window_size)   # ✅ Works
print(config.sampling_rate)         # ❌ AttributeError (audio-only field)
```

### Base Configuration Fields

**All config types inherit from `BaseModelConfig`:**

```python
class BaseModelConfig(BaseModel):
    """Base configuration for all models."""
    
    class_name: str                      # Python class name (discriminator)
    description: str                     # User-friendly name
    model_path: ExistingPath             # Path to weights (validated)
    device: str = "cuda"                 # Target device
    isAudio: bool = False                # Media type flags
    isVideo: bool = True
    isImage: bool = False
    isMultiModal: bool = False
    model_name: Optional[str] = None     # Set by ModelManager
```

**Path Validation:**

```python
# ExistingPath is a custom Pydantic type with validation
ExistingPath = Annotated[Path, AfterValidator(check_path_exists)]

def check_path_exists(path: Path) -> Path:
    """
    Validates that a file path exists.
    Raises ValueError if path doesn't exist or isn't a file.
    """
    if not path.exists():
        raise ValueError(f"File path does not exist: '{path}'")
    if not path.is_file():
        raise ValueError(f"Path is not a file: '{path}'")
    return path

# At startup, Pydantic validates all model_path fields:
config = SiglipLSTMv4Config(model_path="models/missing.pth")
# ❌ ValueError: File path does not exist: 'models/missing.pth'
```

**Media Type Flags:**

```python
# Video detector
class SiglipLSTMv4Config(BaseModelConfig):
    isVideo: bool = True   # Processes video
    isAudio: bool = False
    isImage: bool = False

# Audio detector
class MelSpectrogramCNNConfig(BaseModelConfig):
    isAudio: bool = True   # Processes audio
    isVideo: bool = False
    isImage: bool = False

# Image detector
class DistilDIREv1Config(BaseModelConfig):
    isImage: bool = True   # Processes images
    isVideo: bool = False
    isAudio: bool = False

# Multimodal detector
class MFFMoEV1Config(BaseModelConfig):
    isVideo: bool = True   # Processes video
    isImage: bool = True   # AND images
    isAudio: bool = False
    isMultiModal: bool = True
```

**Used for API routing:**

```python
# API determines which models to use based on media type
@router.post("/analyze")
def analyze_media(file: UploadFile, model_name: str):
    model = registry.get_model(model_name)
    
    # Validate media type compatibility
    if file.content_type.startswith("video/"):
        if not model.config.isVideo:
            raise HTTPException(400, "Model doesn't support video")
    elif file.content_type.startswith("audio/"):
        if not model.config.isAudio:
            raise HTTPException(400, "Model doesn't support audio")
    elif file.content_type.startswith("image/"):
        if not model.config.isImage:
            raise HTTPException(400, "Model doesn't support images")
    
    return model.analyze(file.path)
```

---

## Implementing a New Detector

### Step 1: Create Configuration Schema

```python
# src/config.py

class MyCustomDetectorConfig(BaseModelConfig):
    """
    Configuration for a new custom detector.
    """
    # Discriminator (must match config.yaml)
    class_name: Literal["MyCustomDetector"]
    
    # Media type flags
    isVideo: bool = True
    isAudio: bool = False
    
    # Model-specific fields
    backbone: str                    # e.g., "resnet50"
    input_resolution: int = 224
    num_layers: int = 3
    custom_param: float = 0.5
    
    # Nested architecture config (optional)
    model_definition: Optional[CustomArchConfig] = None
```

### Step 2: Add to ModelConfig Union

```python
# src/config.py

ModelConfig = Annotated[
    Union[
        # ... existing configs
        MyCustomDetectorConfig,  # ← Add here
    ],
    Field(discriminator="class_name"),
]
```

### Step 3: Implement Detector Class

```python
# src/ml/detectors/my_custom_detector.py

import torch
import logging
from typing import Dict, Any

from src.ml.base import BaseModel
from src.app.schemas import VideoAnalysisResult, FramePrediction
from src.config import MyCustomDetectorConfig

logger = logging.getLogger(__name__)


class MyCustomDetector(BaseModel):
    """
    A custom deepfake detector using [describe approach].
    """
    
    def __init__(self, config: MyCustomDetectorConfig):
        """
        Initialize with validated config.
        """
        super().__init__(config)
        self.config: MyCustomDetectorConfig = config  # Type hint for IDE
    
    def load(self) -> None:
        """
        Load custom model architecture and weights.
        """
        logger.info(f"Loading MyCustomDetector from {self.config.model_path}")
        
        # 1. Initialize architecture
        from torchvision.models import resnet50
        model = resnet50(pretrained=False)
        
        # Modify architecture based on config
        model.fc = torch.nn.Linear(
            model.fc.in_features,
            2  # Binary classification
        )
        
        # 2. Load weights
        checkpoint = torch.load(
            self.config.model_path,
            map_location=self.device,
            weights_only=True
        )
        model.load_state_dict(checkpoint)
        
        # 3. Move to device and set eval mode
        self.model = model.to(self.device)
        self.model.eval()
        
        logger.info(f"🚀 MyCustomDetector ready on {self.device}")
    
    def analyze(
        self, 
        media_path: str, 
        generate_visualizations: bool = False, 
        **kwargs
    ) -> VideoAnalysisResult:
        """
        Analyze video with custom approach.
        """
        import time
        start_time = time.time()
        
        # Extract event publishing IDs
        video_id = kwargs.get("video_id")
        user_id = kwargs.get("user_id")
        
        # 1. Preprocess media
        frames = self._extract_frames(media_path)
        
        # 2. Inference loop
        frame_predictions = []
        for i, frame in enumerate(frames):
            # Preprocess frame
            tensor = self._preprocess_frame(frame)
            tensor = tensor.to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.model(tensor)
                probs = torch.softmax(output, dim=1)
                fake_prob = probs[0][1].item()
            
            # Store prediction
            frame_predictions.append(FramePrediction(
                index=i,
                score=fake_prob,
                prediction="FAKE" if fake_prob > 0.5 else "REAL"
            ))
            
            # Publish progress (every 10 frames)
            if (i + 1) % 10 == 0 and video_id and user_id:
                from src.ml.event_publisher import event_publisher
                from src.ml.schemas import ProgressEvent, EventData
                
                event_publisher.publish(ProgressEvent(
                    media_id=video_id,
                    user_id=user_id,
                    event="FRAME_ANALYSIS_PROGRESS",
                    message=f"Analyzed {i + 1}/{len(frames)} frames",
                    data=EventData(
                        model_name=self.config.model_name,
                        progress=i + 1,
                        total=len(frames)
                    )
                ))
        
        # 3. Aggregate results
        import numpy as np
        avg_score = np.mean([p.score for p in frame_predictions])
        final_prediction = "FAKE" if avg_score > 0.5 else "REAL"
        
        # 4. Calculate metrics
        from src.ml.metric_schemas import FrameBasedMetrics
        metrics = FrameBasedMetrics(
            mean_score=float(avg_score),
            std_score=float(np.std([p.score for p in frame_predictions])),
            max_score=float(max(p.score for p in frame_predictions)),
            min_score=float(min(p.score for p in frame_predictions)),
            fake_frame_count=sum(1 for p in frame_predictions if p.prediction == "FAKE"),
            total_frames=len(frame_predictions)
        )
        
        processing_time = time.time() - start_time
        
        # 5. Return structured result
        return VideoAnalysisResult(
            prediction=final_prediction,
            confidence=avg_score if final_prediction == "FAKE" else (1.0 - avg_score),
            processing_time=processing_time,
            frame_count=len(frames),
            frames_analyzed=len(frame_predictions),
            frame_predictions=frame_predictions,
            metrics=metrics,
            visualization_path=None
        )
    
    # Helper methods
    def _extract_frames(self, video_path: str):
        """Extract frames from video."""
        import cv2
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames
    
    def _preprocess_frame(self, frame):
        """Preprocess frame to tensor."""
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config.input_resolution, self.config.input_resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(frame).unsqueeze(0)
```

### Step 4: Add to config.yaml

```yaml
# configs/config.yaml

models:
  MY-CUSTOM-DETECTOR:
    class_name: "MyCustomDetector"  # Must match Literal in config class
    description: "My Custom Deepfake Detector"
    model_path: "models/my-custom-v1.pth"
    device: "cuda"
    isVideo: true
    isAudio: false
    isImage: false
    
    # Model-specific parameters
    backbone: "resnet50"
    input_resolution: 224
    num_layers: 3
    custom_param: 0.7
```

### Step 5: Register in ModelManager

**Automatic discovery** handles this! ModelManager automatically:

1. Reads `config.yaml`
2. Parses model configs with discriminated union
3. Dynamically imports detector class
4. Instantiates detector with validated config

```python
# No manual registration needed!
# Just ensure:
# 1. Class name in config.yaml matches Literal in config schema
# 2. Detector file exists in src/ml/detectors/
# 3. Class implements BaseModel interface

# Usage:
registry.get_model("MY-CUSTOM-DETECTOR")  # Automatically loaded!
```

---

## Type Safety Benefits

### Compile-Time Checking

```python
# With proper type hints, IDEs catch errors:

class MyDetector(BaseModel):
    def load(self) -> None:  # ✅ Correct signature
        pass
    
    def analyze(self, media_path: str) -> int:  # ❌ Wrong return type
        return 42

# IDE/mypy error:
# Return type "int" is incompatible with declared return type "AnalysisResult"
```

### Runtime Validation

```python
# Pydantic validates return values at runtime:

def analyze(self, media_path: str, **kwargs) -> VideoAnalysisResult:
    return VideoAnalysisResult(
        prediction="MAYBE",  # ❌ Invalid (not "REAL" or "FAKE")
        confidence=1.5,      # ❌ Invalid (must be 0.0-1.0)
        processing_time=-1,  # ❌ Invalid (must be positive)
        frames_analyzed=10,
        frame_predictions=[],
        metrics={}
    )

# Pydantic raises ValidationError:
# 1 validation error for VideoAnalysisResult
# prediction
#   Input should be 'REAL' or 'FAKE'
```

### Polymorphic Safety

```python
# Type system ensures all detectors compatible:

def process_media(detector: BaseModel, path: str) -> Dict:
    """
    Works with ANY detector that implements BaseModel.
    """
    # Type-safe calls (guaranteed by ABC)
    detector.load()  # ✅ All detectors have load()
    result = detector.analyze(path)  # ✅ All detectors have analyze()
    
    # result is guaranteed to be AnalysisResult (union type)
    # Can safely access common fields:
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence}")
    
    return {
        "prediction": result.prediction,
        "confidence": result.confidence
    }

# Works for any detector:
process_media(SiglipLSTMV4(...), "video.mp4")
process_media(MelSpectrogramCNNV1(...), "audio.mp3")
process_media(DistilDIREDetectorV1(...), "image.jpg")
```

---

## Design Patterns

### 1. Strategy Pattern

**Definition**: Define a family of algorithms, encapsulate each one, and make them interchangeable.

**Implementation:**

```python
# Context (ModelManager)
class ModelManager:
    def get_model(self, name: str) -> BaseModel:
        """Returns appropriate strategy (detector)."""
        return self._models[name]

# Strategies (Detectors)
class SiglipLSTMV4(BaseModel):  # Strategy 1
    def analyze(self, path): ...

class MelSpectrogramCNNV1(BaseModel):  # Strategy 2
    def analyze(self, path): ...

# Client code
detector = manager.get_model(user_choice)  # Select strategy at runtime
result = detector.analyze(media_path)      # Execute strategy
```

**Benefits:**

- Add new detectors without modifying existing code
- Switch algorithms dynamically based on user input
- Test strategies independently

### 2. Template Method

**Definition**: Define the skeleton of an algorithm, deferring some steps to subclasses.

**Implementation:**

```python
# Template (BaseModel)
class BaseModel(ABC):
    def analyze(self, path, **kwargs) -> AnalysisResult:
        """Template method (skeleton)."""
        # Step 1: Common preprocessing (defined in subclass)
        data = self._preprocess(path)
        
        # Step 2: Model inference (defined in subclass)
        predictions = self._infer(data)
        
        # Step 3: Common postprocessing (defined in subclass)
        result = self._postprocess(predictions)
        
        return result

# Concrete implementations
class SiglipLSTMV4(BaseModel):
    def _preprocess(self, path):
        """Subclass-specific preprocessing."""
        return extract_frames(path)
    
    def _infer(self, frames):
        """Subclass-specific inference."""
        return self.model(frames)
    
    def _postprocess(self, outputs):
        """Subclass-specific postprocessing."""
        return VideoAnalysisResult(...)
```

### 3. Factory Pattern (via ModelManager)

**Definition**: Create objects without specifying their exact classes.

**Implementation:**

```python
# Factory (ModelManager)
class ModelManager:
    def get_model(self, name: str) -> BaseModel:
        """
        Factory method: Creates detector instance
        without client knowing concrete class.
        """
        config = self.settings.models[name]
        
        # Import correct class based on class_name
        module = importlib.import_module(f"src.ml.detectors.{module_name}")
        detector_class = getattr(module, config.class_name)
        
        # Instantiate and return
        return detector_class(config)

# Client code (doesn't know concrete class)
detector = manager.get_model("SIGLIP-LSTM-V4")  # Returns SiglipLSTMV4
result = detector.analyze(path)  # Polymorphic call
```

---

## Summary

BaseModel provides a robust, extensible contract for all detectors:

✅ **Abstract Base Class** - Enforces `load()` and `analyze()` implementation  
✅ **Strategy Pattern** - Interchangeable detection algorithms  
✅ **Type Safety** - Pydantic validation + Python type hints  
✅ **Discriminated Unions** - Automatic config type selection  
✅ **Media Agnostic** - Supports video, audio, image, multimodal  
✅ **Polymorphic Usage** - Treat all models uniformly  
✅ **Extensible** - Add new detectors without modifying existing code  
✅ **Event Integration** - Built-in progress tracking via kwargs  

**Key Integration Points:**

1. **ModelManager** → Loads detectors using Factory pattern
2. **API Routes** → Polymorphically calls `analyze()` on any detector
3. **CLI Tools** → Uses same interface for batch processing
4. **Event Publisher** → Detectors publish progress via kwargs (video_id, user_id)

**Contract Summary:**

```python
class BaseModel(ABC):
    def __init__(self, config: ModelConfig): ...
    
    @abstractmethod
    def load(self) -> None:
        """Load model weights and processors."""
        pass
    
    @abstractmethod
    def analyze(self, media_path: str, generate_visualizations: bool = False, **kwargs) -> AnalysisResult:
        """Perform inference and return structured results."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {...}
```

**All detectors must implement this contract to participate in the system.**
