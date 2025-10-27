# Configuration System

## Overview

The **Configuration System** (`src/config.py`) provides type-safe, validated configuration management using **Pydantic** for runtime validation and YAML for human-readable configuration. It combines `config.yaml` (application settings) and `.env` (secrets/environment) into a unified, validated `Settings` object.

**Purpose:** Ensure configuration correctness at startup, prevent runtime errors from misconfiguration, and provide flexible deployment options.

**Key Technologies:**

- **Pydantic v2**: Runtime validation, type coercion, discriminated unions
- **YAML**: Human-readable model configurations
- **python-dotenv**: Environment variable loading
- **Annotated Types**: Custom validators for path existence

---

## Core Concepts

### Configuration Layers

```text
┌─────────────────────────────────────────────────────────────┐
│                   Configuration Layers                       │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Layer 1: config.yaml (Application Settings)           │ │
│  │  - Model configurations                                │ │
│  │  - Project name                                        │ │
│  │  - Model-specific parameters                           │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ▼                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Layer 2: .env (Secrets & Environment)                 │ │
│  │  - API_KEY                                             │ │
│  │  - DEVICE (cuda/cpu)                                   │ │
│  │  - ACTIVE_MODELS                                       │ │
│  │  - DEFAULT_MODEL_NAME                                  │ │
│  │  - STORAGE_PATH, ASSETS_BASE_URL                       │ │
│  │  - REDIS_URL                                           │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ▼                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Layer 3: Pydantic Validation                          │ │
│  │  - Type coercion (str → Path, JSON → List)            │ │
│  │  - Discriminated union resolution                      │ │
│  │  - Path existence validation                           │ │
│  │  - Field validation (@model_validator)                 │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ▼                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Output: Validated Settings Object                     │ │
│  │  - settings.models: Dict[str, ModelConfig]            │ │
│  │  - settings.active_models: List[str]                  │ │
│  │  - settings.device: str                               │ │
│  │  - All paths validated to exist                       │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Loading Flow

**Server Mode (Web API):**

```python
# src/config.py (bottom of file)
try:
    settings = Settings.from_yaml_and_env("configs/config.yaml")
except (RuntimeError, ValueError, ValidationError) as e:
    logger.critical(f"❌ FATAL: Could not load configuration. Server cannot start.\nError: {e}")
    sys.exit(1)
```

**CLI Mode:**

```python
# CLI tools use different loading method
settings = Settings.from_yaml_and_env_cli("configs/config.yaml")
# Difference: Loads ALL models from config.yaml, not just ACTIVE_MODELS
```

**Flow:**

```text
1. Load .env file → Environment variables
2. Read config.yaml → Dict
3. Parse ACTIVE_MODELS from env (JSON array or comma-separated)
4. Merge config.yaml + .env overrides
5. Filter models dict to only ACTIVE_MODELS (server mode)
6. Pydantic validates entire Settings object:
   a. Discriminated union selects correct ModelConfig type
   b. ExistingPath validators check file existence
   c. @model_validator runs cross-field validation
7. If validation fails → Log error + exit(1)
8. If validation succeeds → settings object ready
```

---

## Environment Variables

### .env File Structure

```bash
# .env

# ===================================================================
# SECURITY
# ===================================================================
API_KEY=your_secret_api_key_here_change_in_production

# ===================================================================
# MODEL CONFIGURATION
# ===================================================================

# Which model to use by default when no model is specified
DEFAULT_MODEL_NAME=SIGLIP-LSTM-V4

# Models to load at server startup (JSON array or comma-separated)
# JSON format (recommended):
ACTIVE_MODELS=["SIGLIP-LSTM-V4","MEL-SPECTROGRAM-CNN-V2","DISTIL-DIRE-V1"]

# OR comma-separated format:
# ACTIVE_MODELS=SIGLIP-LSTM-V4,MEL-SPECTROGRAM-CNN-V2,DISTIL-DIRE-V1

# ===================================================================
# HARDWARE
# ===================================================================

# Device to run models on: "cuda" or "cpu"
DEVICE=cuda

# ===================================================================
# STORAGE & ASSETS
# ===================================================================

# Base URL for serving media files (visualizations, spectrograms)
ASSETS_BASE_URL=http://localhost:3000

# Local storage path for generated files
STORAGE_PATH=../Backend/public/media

# ===================================================================
# REDIS (OPTIONAL - For Real-Time Progress)
# ===================================================================

# Redis connection URL for event publishing
REDIS_URL=redis://localhost:6379

# Redis channel name for progress events
MEDIA_PROGRESS_CHANNEL_NAME=media:progress
```

### ACTIVE_MODELS Parsing

**Supports two formats:**

```python
# Format 1: JSON array (recommended)
ACTIVE_MODELS=["SIGLIP-LSTM-V4","MEL-SPECTROGRAM-CNN-V2"]

# Format 2: Comma-separated (backward compatible)
ACTIVE_MODELS=SIGLIP-LSTM-V4,MEL-SPECTROGRAM-CNN-V2
```

**Parsing Logic:**

```python
active_models_str = os.getenv("ACTIVE_MODELS", "")
active_model_names = []

if active_models_str:
    active_models_str = active_models_str.strip()
    
    # Try JSON array first
    if active_models_str.startswith('[') and active_models_str.endswith(']'):
        try:
            active_model_names = json.loads(active_models_str)
            logger.info(f"🔧 Parsed ACTIVE_MODELS as JSON array: {len(active_model_names)} models")
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse ACTIVE_MODELS as JSON, falling back to comma-separated")
            active_model_names = [m.strip() for m in active_models_str.split(',') if m.strip()]
    else:
        # Parse as comma-separated
        active_model_names = [m.strip() for m in active_models_str.split(',') if m.strip()]
        logger.info(f"🔧 Parsed ACTIVE_MODELS as comma-separated: {len(active_model_names)} models")
```

**Why Two Formats?**

- **JSON Array**: Unambiguous, supports model names with commas, better for complex deployments
- **Comma-Separated**: Simpler, backward compatible, easier for quick testing

---

## Settings Class

### Main Settings Schema

```python
class Settings(BaseSettings):
    """
    Main application settings loaded from config.yaml and .env.
    Validated by Pydantic at runtime.
    """
    # Security
    api_key: SecretStr                           # From .env (never logged)
    
    # Model configuration
    default_model_name: str                      # From .env
    active_models: List[str] = Field(default_factory=list)  # From .env (parsed)
    device: str                                  # From .env ("cuda" or "cpu")
    project_name: str                            # From config.yaml
    models: Dict[str, ModelConfig]               # From config.yaml (validated union)
    
    # Storage & Assets
    assets_base_url: str = Field(..., alias="ASSETS_BASE_URL")  # From .env
    storage_path: Path = Field(..., alias="STORAGE_PATH")        # From .env
    
    # Redis (Optional)
    redis_url: Optional[str] = Field(None, alias="REDIS_URL")
    media_progress_channel_name: str = Field("media-progress-events", alias="MEDIA_PROGRESS_CHANNEL_NAME")
    
    # Pydantic configuration
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'  # Ignore unknown env vars
    )
```

**Field Aliases:**

```python
# Without alias:
STORAGE_PATH=./storage  # Won't work (Pydantic expects lowercase)

# With alias:
storage_path: Path = Field(..., alias="STORAGE_PATH")
# Now both work:
# - STORAGE_PATH=./storage (env var, uppercase)
# - storage_path: "./storage" (YAML, lowercase)
```

### Cross-Field Validation

```python
@model_validator(mode='after')
def validate_paths_and_models(self) -> 'Settings':
    """
    Validates configuration after all fields are parsed.
    Runs after individual field validators.
    """
    # 1. Validate default model is in active models
    if self.active_models and self.default_model_name not in self.active_models:
        raise ValueError(
            f"Config error: Default model '{self.default_model_name}' not in ACTIVE_MODELS."
        )
    
    # 2. Validate and create storage path
    if not self.storage_path.exists():
        logger.info(f"Storage path '{self.storage_path}' not found. Creating it.")
        self.storage_path.mkdir(parents=True, exist_ok=True)
    elif not self.storage_path.is_dir():
        raise ValueError(f"Config error: STORAGE_PATH '{self.storage_path}' must be a directory.")
    
    return self
```

**Why Cross-Field Validation?**

```python
# Individual field validation:
default_model_name: str  # ✅ Type is str
active_models: List[str]  # ✅ Type is List[str]

# But need to check relationship:
# ❌ default_model_name="INVALID-MODEL"
# ✅ active_models=["SIGLIP-LSTM-V4"]
# Problem: Default model not in active models!

# Cross-field validator catches this:
@model_validator(mode='after')
def validate_paths_and_models(self):
    if self.default_model_name not in self.active_models:
        raise ValueError("...")  # ❌ Caught at startup!
```

---

## Model Configuration

### Discriminated Union

**ModelConfig is a union of all model-specific config types:**

```python
ModelConfig = Annotated[
    Union[
        ColorCuesConfig,
        SiglipLSTMv1Config,
        SiglipLSTMv3Config,
        SiglipLSTMv4Config,
        EfficientNetB7Config,
        EyeblinkModelConfig,
        ScatteringWaveV1Config,
        MelSpectrogramCNNConfig,
        MelSpectrogramCNNv3Config,
        STFTSpectrogramCNNConfig,
        DistilDIREv1Config,
        MFFMoEV1Config,
        CrossEfficientViTConfig,
        LipFDv1Config
    ],
    Field(discriminator="class_name"),  # ← Discriminator field
]
```

**How Discrimination Works:**

```yaml
# config.yaml
models:
  SIGLIP-LSTM-V4:
    class_name: "SiglipLSTMV4"  # ← Pydantic uses this to select config type
    description: "V4 SigLIP-LSTM..."
    model_path: "models/SigLip-LSTM-v4.pth"
    processor_path: "google/siglip-base-patch16-224"
    num_frames: 120
    rolling_window_size: 10
    model_definition:
      base_model_path: "google/siglip-base-patch16-224"
      lstm_hidden_size: 512
      lstm_num_layers: 2
      num_classes: 1
      dropout_rate: 0.5
```

**Pydantic's Resolution:**

```python
# Step 1: Parse YAML
raw_config = {
    "class_name": "SiglipLSTMV4",
    "description": "...",
    ...
}

# Step 2: Check discriminator field
discriminator_value = raw_config["class_name"]  # "SiglipLSTMV4"

# Step 3: Find matching config in union
for config_type in [ColorCuesConfig, SiglipLSTMv1Config, ...]:
    if config_type.model_fields["class_name"].annotation == Literal["SiglipLSTMV4"]:
        selected_type = config_type  # SiglipLSTMv4Config
        break

# Step 4: Validate with selected type
validated_config = SiglipLSTMv4Config(**raw_config)

# Result: Type-safe, fully validated config object
assert isinstance(validated_config, SiglipLSTMv4Config)
assert validated_config.processor_path == "google/siglip-base-patch16-224"
assert validated_config.rolling_window_size == 10
```

### Base Model Configuration

**All model configs inherit from `BaseModelConfig`:**

```python
class BaseModelConfig(BaseModel):
    """Base configuration for all models."""
    
    # Discriminator (must match Python class name)
    class_name: str
    
    # Common fields
    description: str                             # User-friendly name
    model_path: ExistingPath                     # Path to weights (validated)
    device: str = "cuda"                         # Target device
    
    # Media type flags
    isAudio: bool = False
    isVideo: bool = True
    isImage: bool = False
    isMultiModal: bool = False
    
    # Runtime field (set by ModelManager)
    model_name: Optional[str] = None
```

**ExistingPath Validator:**

```python
# Custom Pydantic type with validation
ExistingPath = Annotated[Path, AfterValidator(check_path_exists)]

def check_path_exists(path: Path) -> Path:
    """
    Validates that a file path exists.
    Called automatically by Pydantic after type coercion.
    """
    if not path.exists():
        raise ValueError(f"Configuration error: File path does not exist: '{path}'")
    if not path.is_file():
        raise ValueError(f"Configuration error: Path is not a file: '{path}'")
    return path

# Usage in config:
class SiglipLSTMv4Config(BaseModelConfig):
    model_path: ExistingPath  # ← Automatically validated at parse time

# At startup:
config = SiglipLSTMv4Config(model_path="models/missing.pth")
# ❌ ValueError: Configuration error: File path does not exist: 'models/missing.pth'
# Server won't start with invalid paths!
```

**ExistingModelDir Validator (for directories):**

```python
# Used by MFF-MoE which has a directory instead of single file
ExistingModelDir = Annotated[Path, AfterValidator(check_model_dir_exists)]

def check_model_dir_exists(path: Path) -> Path:
    """Validates model directory and required files."""
    logger.info(f"MFF-MoE-v1 Models Folder Path: {path}")
    
    if not path.is_dir():
        raise ValueError(f"Configuration error: Path is not a directory: '{path}'")
    
    # Check for required files
    weight_file = path / "MFF-MoE-v1.pth"
    state_file = path / "MFF-MoE-v1.state"
    
    if not weight_file.exists() or not state_file.exists():
        raise ValueError(
            f"Configuration error: Directory '{path}' must contain "
            "'MFF-MoE-v1.pth' and 'MFF-MoE-v1.state'"
        )
    
    return path

# Usage:
class MFFMoEV1Config(BaseModelConfig):
    model_path: ExistingModelDir  # ← Validates directory + required files
```

### Model-Specific Configurations

**Video Model Example (SiglipLSTMv4Config):**

```python
class SiglipLSTMv4Config(BaseModelConfig):
    """Configuration for SIGLIP-LSTM-V4 detector."""
    
    # Discriminator (MUST match Python class name)
    class_name: Literal["SiglipLSTMV4"]
    
    # Model-specific fields
    processor_path: str                          # HuggingFace processor
    num_frames: int                              # Frames to extract
    rolling_window_size: int                     # Window size for analysis
    
    # Nested architecture config
    model_definition: SiglipArchitectureV4Config
```

**Audio Model Example (MelSpectrogramCNNConfig):**

```python
class MelSpectrogramCNNConfig(BaseModelConfig):
    """Configuration for Mel-Spectrogram-CNN detector."""
    
    # Supports both V1 and V2 (non-overlapping)
    class_name: Literal["MelSpectrogramCNNV1", "MelSpectrogramCNNV2"]
    
    # Audio-specific fields
    sampling_rate: int                           # 16000 Hz
    n_fft: int                                   # FFT window size
    hop_length: int                              # Hop between windows
    n_mels: int                                  # Number of mel bands
    dpi: int                                     # Spectrogram image DPI
    chunk_duration_s: float                      # Audio chunk duration
```

**Audio Model with Overlap (MelSpectrogramCNNv3Config):**

```python
class MelSpectrogramCNNv3Config(BaseModelConfig):
    """Configuration for Mel-Spectrogram-CNN-V3 with overlapping chunks."""
    
    class_name: Literal["MelSpectrogramCNNV3"]
    
    # Inherits all fields from MelSpectrogramCNNConfig
    sampling_rate: int
    n_fft: int
    hop_length: int
    n_mels: int
    dpi: int
    chunk_duration_s: float
    
    # V3-specific: overlapping chunks
    chunk_overlap_s: float  # ← NEW in V3
```

**Image Model Example (DistilDIREv1Config):**

```python
class DistilDIREv1Config(BaseModelConfig):
    """Configuration for DistilDIRE image detector."""
    
    class_name: Literal["DistilDIREDetectorV1"]
    
    # Image-specific fields
    adm_model_path: ExistingPath                 # ADM diffusion model path
    image_size: int = 256                        # Input image size
    adm_config: Dict[str, Any] = Field(default_factory=dict)  # ADM parameters
    
    # Override base class defaults
    isImage: bool = True
    isVideo: bool = False
```

**Multimodal Model Example (MFFMoEV1Config):**

```python
class MFFMoEV1Config(BaseModelConfig):
    """Configuration for MFF-MoE multimodal detector."""
    
    class_name: Literal["MFFMoEDetectorV1"]
    
    # Directory instead of single file
    model_path: ExistingModelDir                 # Validated directory
    
    # Model-specific
    video_frames_to_sample: int = 100
    
    # Multimodal flags
    isImage: bool = True
    isVideo: bool = True
    isAudio: bool = False
    isMultiModal: bool = True
```

### Architecture Configurations

**Nested configuration schemas for model architectures:**

```python
class SiglipArchitectureV4Config(BaseModel):
    """Architecture definition for SIGLIP-LSTM-V4."""
    
    base_model_path: str                         # HuggingFace model
    lstm_hidden_size: int                        # LSTM hidden units
    lstm_num_layers: int                         # Number of LSTM layers
    num_classes: int                             # Output classes
    dropout_rate: float = 0.5                    # Dropout probability
```

**With Field Aliases (for hyphenated YAML keys):**

```python
class CrossEfficientViTArchConfig(BaseModel):
    """Architecture for Cross-EfficientNet-ViT detector."""
    
    # YAML uses hyphenated keys, Python uses underscores
    image_size: int = Field(..., alias='image-size')
    num_classes: int = Field(..., alias='num-classes')
    depth: int
    sm_dim: int = Field(..., alias='sm-dim')
    sm_patch_size: int = Field(..., alias='sm-patch-size')
    # ... 20+ fields with aliases
```

**YAML config:**

```yaml
model_definition:
  image-size: 224          # ← Hyphenated in YAML
  num-classes: 1
  sm-dim: 192
  sm-patch-size: 7
```

**Parsed to Python:**

```python
config.model_definition.image_size  # ✅ 224 (underscore in Python)
config.model_definition.num_classes  # ✅ 1
config.model_definition.sm_dim      # ✅ 192
```

---

## Loading Methods

### from_yaml_and_env (Server Mode)

**Used by web API server:**

```python
@classmethod
def from_yaml_and_env(cls, yaml_path: str, env_file: str = '.env') -> "Settings":
    """
    Load settings for server mode.
    Only includes models specified in ACTIVE_MODELS environment variable.
    """
    from dotenv import load_dotenv
    load_dotenv(env_file)
    
    # 1. Load environment variables
    env_device = os.getenv('DEVICE', 'cpu').lower()
    default_model_name_env = os.getenv("DEFAULT_MODEL_NAME")
    active_models_str = os.getenv("ACTIVE_MODELS", "")
    
    # 2. Parse ACTIVE_MODELS (JSON or comma-separated)
    active_model_names = parse_active_models(active_models_str)
    
    # 3. Load YAML
    with open(yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file) or {}
    
    # 4. Merge env overrides
    env_data = {
        "api_key": os.getenv("API_KEY"),
        "default_model_name": default_model_name_env.strip() if default_model_name_env else None,
        "active_models": active_model_names,
        "device": env_device,
        "ASSETS_BASE_URL": os.getenv("ASSETS_BASE_URL"),
        "STORAGE_PATH": os.getenv("STORAGE_PATH", "../Backend/public/media"),
    }
    
    merged_config = {**yaml_data, **{k: v for k, v in env_data.items() if v is not None}}
    
    # 5. Set device for all models
    if merged_config.get('models'):
        for model_name in merged_config['models']:
            if isinstance(merged_config['models'][model_name], dict):
                merged_config['models'][model_name]['device'] = env_device
    
    # 6. FILTER: Only include ACTIVE_MODELS
    if 'models' in merged_config:
        merged_config['models'] = {
            name: config for name, config in merged_config['models'].items()
            if name in active_model_names
        }
    
    # 7. Validate and return
    return cls(**merged_config)
```

**Effect of ACTIVE_MODELS Filtering:**

```yaml
# config.yaml has 15 models
models:
  SIGLIP-LSTM-V4: ...
  MEL-SPECTROGRAM-CNN-V2: ...
  DISTIL-DIRE-V1: ...
  # ... 12 more models
```

```bash
# .env specifies only 2 active models
ACTIVE_MODELS=["SIGLIP-LSTM-V4","MEL-SPECTROGRAM-CNN-V2"]
```

```python
# Result: settings.models contains only 2 models
settings.models = {
    "SIGLIP-LSTM-V4": SiglipLSTMv4Config(...),
    "MEL-SPECTROGRAM-CNN-V2": MelSpectrogramCNNConfig(...)
}
# Other 13 models not loaded (saves memory)
```

### from_yaml_and_env_cli (CLI Mode)

**Used by CLI tools:**

```python
@classmethod
def from_yaml_and_env_cli(cls, yaml_path: str, env_file: str = '.env') -> "Settings":
    """
    Load settings for CLI mode.
    Includes ALL models from config.yaml, not just ACTIVE_MODELS.
    """
    from dotenv import load_dotenv
    load_dotenv(env_file)
    
    # ... similar to server mode ...
    
    # KEY DIFFERENCE: Include ALL models from config.yaml
    all_model_names = list(yaml_data.get('models', {}).keys())
    
    env_data = {
        "api_key": os.getenv("API_KEY", "cli-mode-key"),  # CLI doesn't need API key
        "default_model_name": default_model_name_env or list(yaml_data['models'].keys())[0],
        "active_models": all_model_names,  # ← ALL models, not just ACTIVE_MODELS
        "device": env_device,
        "ASSETS_BASE_URL": os.getenv("ASSETS_BASE_URL", "http://localhost:3000"),
        "STORAGE_PATH": os.getenv("STORAGE_PATH", "./storage"),
    }
    
    # NO FILTERING: All models included
    logger.info(f"🔧 [CLI MODE] Loaded {len(merged_config.get('models', {}))} models from config")
    
    return cls(**merged_config)
```

**Why Two Loading Methods?**

| Feature | Server Mode | CLI Mode |
|---------|-------------|----------|
| **Models Loaded** | Only ACTIVE_MODELS | All models from config.yaml |
| **API Key** | Required | Optional (defaults to "cli-mode-key") |
| **Use Case** | Production web server | Development, testing, batch processing |
| **Memory Usage** | Low (only active models) | Higher (all models available) |
| **Flexibility** | Controlled (ops team sets ACTIVE_MODELS) | Full (developers can use any model) |

**CLI Usage Example:**

```bash
# CLI can use any model from config.yaml
python -m src.cli.predict video.mp4 --model CROSS-EFFICIENT-VIT-GAN

# Even if ACTIVE_MODELS doesn't include it
# (because CLI loads all models)
```

---

## Configuration Validation

### Startup Validation

**At server startup:**

```python
# src/config.py (bottom of file)
try:
    settings = Settings.from_yaml_and_env("configs/config.yaml")
except (RuntimeError, ValueError, ValidationError) as e:
    logger.critical(f"❌ FATAL: Could not load configuration. Server cannot start.\nError: {e}")
    import sys
    sys.exit(1)
```

**What Gets Validated:**

1. **File Existence**: All `model_path` fields checked
2. **Type Correctness**: Strings are strings, ints are ints, etc.
3. **Discriminator Match**: `class_name` matches one of the union types
4. **Field Requirements**: All required fields present
5. **Cross-Field Logic**: Default model in active models, storage path is directory
6. **Nested Validation**: Architecture configs validated recursively

**Validation Failure Example:**

```yaml
# config.yaml
models:
  SIGLIP-LSTM-V4:
    class_name: "INVALID_CLASS_NAME"  # ❌ Not in discriminated union
    model_path: "models/missing.pth"   # ❌ File doesn't exist
    num_frames: "not_a_number"         # ❌ Should be int
```

```text
❌ FATAL: Could not load configuration. Server cannot start.
Error: 3 validation errors for Settings
models.SIGLIP-LSTM-V4.class_name
  Input should be 'SiglipLSTMV4' [type=literal_error]
models.SIGLIP-LSTM-V4.model_path
  Configuration error: File path does not exist: 'models/missing.pth' [type=value_error]
models.SIGLIP-LSTM-V4.num_frames
  Input should be a valid integer [type=int_type]
```

**Server exits with code 1** (container orchestrators can detect failure)

### Runtime Access

**Type-safe access throughout application:**

```python
# Import validated settings
from src.config import settings

# Access with full IDE autocomplete and type checking
print(settings.device)                    # "cuda" or "cpu"
print(settings.default_model_name)        # "SIGLIP-LSTM-V4"
print(settings.active_models)             # ["SIGLIP-LSTM-V4", ...]

# Get model config (type-safe)
model_config = settings.models["SIGLIP-LSTM-V4"]
assert isinstance(model_config, SiglipLSTMv4Config)

# Access model-specific fields
print(model_config.processor_path)        # "google/siglip-base-patch16-224"
print(model_config.rolling_window_size)   # 10

# Secrets are protected
print(settings.api_key)                   # SecretStr('**********')
print(settings.api_key.get_secret_value())  # "your_secret_key"
```

---

## Example Configurations

### Production Deployment

**.env (Production):**

```bash
# Production settings
API_KEY=prod_secret_key_change_me_xyz789

DEFAULT_MODEL_NAME=SIGLIP-LSTM-V4

# Load only production models (lightweight)
ACTIVE_MODELS=["SIGLIP-LSTM-V4","MEL-SPECTROGRAM-CNN-V3","MFF-MOE-V1"]

# GPU deployment
DEVICE=cuda

# Production URLs
ASSETS_BASE_URL=https://api.drishtiksha.ai
STORAGE_PATH=/var/app/storage

# Redis for production
REDIS_URL=redis://redis-cluster:6379
MEDIA_PROGRESS_CHANNEL_NAME=media:progress
```

### Development Environment

**.env (Development):**

```bash
# Development settings
API_KEY=dev_insecure_key_for_testing

DEFAULT_MODEL_NAME=SIGLIP-LSTM-V4

# Load all models for testing
ACTIVE_MODELS=["SIGLIP-LSTM-V4","SIGLIP-LSTM-V3","MEL-SPECTROGRAM-CNN-V2","MEL-SPECTROGRAM-CNN-V3","DISTIL-DIRE-V1"]

# CPU for local development (no GPU)
DEVICE=cpu

# Local URLs
ASSETS_BASE_URL=http://localhost:3000
STORAGE_PATH=../Backend/public/media

# Local Redis
REDIS_URL=redis://localhost:6379
```

### Docker Deployment

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  ml-server:
    build: ./Server
    environment:
      - API_KEY=${API_KEY}
      - DEFAULT_MODEL_NAME=${DEFAULT_MODEL_NAME}
      - ACTIVE_MODELS=["SIGLIP-LSTM-V4","MEL-SPECTROGRAM-CNN-V3"]
      - DEVICE=cuda
      - ASSETS_BASE_URL=http://localhost:3000
      - STORAGE_PATH=/app/storage
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./models:/app/models
      - ./storage:/app/storage
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

---

## Best Practices

### 1. Never Commit Secrets

```bash
# ❌ WRONG: Committing .env to git
git add .env
git commit -m "Add config"

# ✅ RIGHT: Use .env.example template
# .env.example (template with placeholders)
API_KEY=your_secret_key_here
DEFAULT_MODEL_NAME=SIGLIP-LSTM-V4
DEVICE=cuda

# .gitignore
.env
```

### 2. Validate Configuration Early

```python
# ✅ RIGHT: Validate at import time (module level)
# src/config.py
try:
    settings = Settings.from_yaml_and_env("configs/config.yaml")
except ValidationError as e:
    logger.critical(f"Configuration error: {e}")
    sys.exit(1)

# ❌ WRONG: Lazy validation (fails at runtime)
settings = None
def get_settings():
    global settings
    if not settings:
        settings = Settings.from_yaml_and_env("configs/config.yaml")
    return settings
```

### 3. Use Type Hints

```python
# ✅ RIGHT: Type hints everywhere
def load_model(config: SiglipLSTMv4Config) -> torch.nn.Module:
    processor_path = config.processor_path  # IDE knows this is str
    num_frames = config.num_frames          # IDE knows this is int
    ...

# ❌ WRONG: No type hints (runtime errors)
def load_model(config):
    processor_path = config.processor_path  # What type? IDE doesn't know
    ...
```

### 4. Document Configuration Changes

```yaml
# config.yaml

models:
  SIGLIP-LSTM-V4:
    class_name: "SiglipLSTMV4"
    # CHANGED 2025-10-26: Increased from 8 to 10 for better temporal context
    rolling_window_size: 10
    # ADDED 2025-10-20: Dropout for regularization
    model_definition:
      dropout_rate: 0.5
```

### 5. Use Discriminated Unions for Extensibility

```python
# ✅ RIGHT: Add new model type to union
ModelConfig = Annotated[
    Union[
        # ... existing configs
        MyNewDetectorConfig,  # ← Just add to union
    ],
    Field(discriminator="class_name"),
]

# Pydantic automatically handles it!
```

---

## Troubleshooting

### Common Errors

**1. Discriminator Mismatch:**

```text
Error: 1 validation error for Settings
models.SIGLIP-LSTM-V4
  Unable to extract tag using discriminator 'class_name'
```

**Solution:** Ensure `class_name` in YAML matches Literal in config class

```yaml
# config.yaml
class_name: "SiglipLSTMV4"  # ← Must match exactly

# src/config.py
class SiglipLSTMv4Config(BaseModelConfig):
    class_name: Literal["SiglipLSTMV4"]  # ← Same spelling
```

**2. File Not Found:**

```text
Error: 1 validation error for Settings
models.SIGLIP-LSTM-V4.model_path
  Configuration error: File path does not exist: 'models/SigLip-LSTM-v4.pth'
```

**Solution:** Check file path and ensure file exists

```bash
ls -la models/SigLip-LSTM-v4.pth
# If missing, download or check path in config.yaml
```

**3. ACTIVE_MODELS Parse Error:**

```text
Error: ValueError: Config error: Default model 'SIGLIP-LSTM-V4' not in ACTIVE_MODELS.
```

**Solution:** Ensure DEFAULT_MODEL_NAME is in ACTIVE_MODELS

```bash
# .env
DEFAULT_MODEL_NAME=SIGLIP-LSTM-V4
ACTIVE_MODELS=["SIGLIP-LSTM-V4","MEL-SPECTROGRAM-CNN-V2"]  # ← Includes default
```

**4. Type Coercion Failure:**

```text
Error: 1 validation error for Settings
models.SIGLIP-LSTM-V4.num_frames
  Input should be a valid integer, unable to parse string as an integer
```

**Solution:** Fix type in YAML

```yaml
# ❌ WRONG
num_frames: "120"  # String

# ✅ RIGHT
num_frames: 120    # Integer
```

---

## Summary

The Configuration System provides:

✅ **Type Safety** - Pydantic validates all configs at startup  
✅ **Discriminated Unions** - Automatic config type selection  
✅ **Path Validation** - Checks file existence before server starts  
✅ **Flexible Deployment** - YAML + .env for different environments  
✅ **Cross-Field Validation** - Ensures configuration consistency  
✅ **Secrets Management** - SecretStr for sensitive data  
✅ **Dual Loading Modes** - Server (filtered) vs CLI (all models)  
✅ **Environment Overrides** - .env overrides config.yaml  
✅ **Fail-Fast** - Invalid config → exit(1) at startup  

**Key Integration Points:**

1. **Startup** → Settings.from_yaml_and_env() → Validated settings object
2. **ModelManager** → settings.models → Type-safe model configs
3. **API Routes** → settings.api_key → Secure authentication
4. **Dependencies** → settings.storage_path → File operations
5. **Event Publisher** → settings.redis_url → Real-time progress

**Configuration Flow:**

```text
config.yaml + .env
      ↓
Load & Merge
      ↓
Pydantic Validation
      ↓
Discriminated Union Resolution
      ↓
Path Existence Checks
      ↓
Cross-Field Validation
      ↓
Validated Settings Object
      ↓
Runtime Type Safety
```

**Production Checklist:**

- [ ] Set strong API_KEY in .env
- [ ] Configure ACTIVE_MODELS for production workload
- [ ] Set DEVICE=cuda for GPU deployment
- [ ] Update ASSETS_BASE_URL to production domain
- [ ] Configure Redis for real-time progress
- [ ] Validate all model_path files exist
- [ ] Test configuration with validation errors
- [ ] Monitor startup logs for config issues
