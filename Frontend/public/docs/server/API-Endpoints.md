# API Endpoints & Schemas

## Overview

The **FastAPI REST API** (`src/app/`) provides HTTP endpoints for deepfake detection inference, model management, and system monitoring. Built on FastAPI 0.100+, it offers automatic OpenAPI documentation, async request handling, and robust error handling.

**Base URL:** `http://localhost:8000` (configurable via environment)

**API Version:** 3.0.0

---

## Core Concepts

### Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                        Client                                │
│                (Frontend, CLI, External)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP Requests
                       │ (X-API-Key header)
┌──────────────────────▼──────────────────────────────────────┐
│                    FastAPI Server                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         Middleware Stack                               │ │
│  │  1. Correlation ID Middleware (request tracing)        │ │
│  │  2. Exception Handlers (global error handling)         │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         Security Layer                                 │ │
│  │  - API Key Authentication (X-API-Key header)           │ │
│  │  - Constant-time comparison (timing attack prevention) │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         Router Layer                                   │ │
│  │  ┌──────────────┐  ┌──────────────┐                   │ │
│  │  │  Analysis    │  │   Status     │                   │ │
│  │  │   Router     │  │   Router     │                   │ │
│  │  │  /analyze    │  │  /, /stats   │                   │ │
│  │  └──────────────┘  └──────────────┘                   │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         Dependencies                                   │ │
│  │  - ModelManager (singleton)                            │ │
│  │  - Media Processing (multipart/form-data handling)     │ │
│  │  - Model Selection (auto or explicit)                  │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              ModelManager + Detectors                        │
│         (Business logic, ML inference)                       │
└─────────────────────────────────────────────────────────────┘
```

### Request Flow

**Typical Analysis Request:**

1. **Client** sends multipart/form-data POST to `/analyze`
2. **Correlation ID Middleware** generates/extracts request ID
3. **Security Layer** validates X-API-Key header
4. **Dependencies** process uploaded media file
5. **Model Selection** chooses appropriate detector
6. **Analysis Router** calls detector's `analyze()` method
7. **Response** returns structured Pydantic model as JSON

---

## Authentication

### API Key Header

**All protected endpoints require API key authentication:**

```http
X-API-Key: your_secret_api_key_here
```

**Configuration:**

```bash
# .env file
API_KEY=your_secret_api_key_here
```

**Security Implementation:**

```python
# src/app/security.py

def get_api_key(
    request: Request,
    api_key_header: str = Security(APIKeyHeader(name="X-API-Key"))
):
    """
    Validates API key using constant-time comparison.
    Prevents timing attacks by using secrets.compare_digest().
    """
    correct_api_key = settings.api_key.get_secret_value()
    client_ip = request.client.host if request.client else "unknown"

    if api_key_header and secrets.compare_digest(api_key_header, correct_api_key):
        return api_key_header  # Valid key
    else:
        logger.warning(
            f"Unauthorized API access attempt from IP: {client_ip}. "
            f"Reason: {'Missing' if not api_key_header else 'Invalid'} API Key."
        )
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API Key"
        )
```

**Why Constant-Time Comparison?**

```python
# ❌ INSECURE: Vulnerable to timing attacks
if api_key_header == correct_api_key:
    ...

# ✅ SECURE: Constant-time comparison
if secrets.compare_digest(api_key_header, correct_api_key):
    ...
```

Timing attacks exploit the fact that string comparison stops at the first mismatch:

- `"abc123" == "xyz789"` → Fails immediately (fast)
- `"abc123" == "abc789"` → Fails at 4th character (slower)

Attackers can measure response times to infer correct characters. `secrets.compare_digest()` always takes the same time regardless of input.

### Public vs Protected Endpoints

```python
# Public endpoints (no authentication)
public_router = APIRouter(tags=["Status & Statistics"])

@public_router.get("/")
def get_root_health():  # No authentication required
    ...

# Protected endpoints (require API key)
private_router = APIRouter(
    tags=["Status & Statistics"],
    dependencies=[Depends(get_api_key)]  # Auth applied to all routes
)

@private_router.get("/stats")
def get_server_stats():  # Requires X-API-Key header
    ...
```

---

## Analysis Endpoints

### POST /analyze

**Perform deepfake detection analysis on uploaded media.**

**Authentication:** Required (X-API-Key)

**Request:**

```http
POST /analyze HTTP/1.1
Host: localhost:8000
X-API-Key: your_secret_key
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary

------WebKitFormBoundary
Content-Disposition: form-data; name="media"; filename="video.mp4"
Content-Type: video/mp4

<binary video data>
------WebKitFormBoundary
Content-Disposition: form-data; name="model"

SIGLIP-LSTM-V4
------WebKitFormBoundary
Content-Disposition: form-data; name="mediaId"

abc123-uuid
------WebKitFormBoundary
Content-Disposition: form-data; name="userId"

user456-uuid
------WebKitFormBoundary--
```

**Form Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `media` | File | ✅ | Media file to analyze (video/audio/image) |
| `model` | String | ❌ | Model name (e.g., "SIGLIP-LSTM-V4"). If omitted, auto-selects best model |
| `mediaId` | String | ❌ | Unique media ID for event publishing |
| `userId` | String | ❌ | User ID for event publishing |

**Model Selection Logic:**

```text
If model specified:
    ✅ Use specified model (validates compatibility)

Else (auto-select):
    1. Determine media type (video/audio/image)
    
    2. Find SPECIALIST models:
       - Image: isImage=True, isVideo=False, isAudio=False
       - Video: isVideo=True, isImage=False, isAudio=False
       - Audio: isAudio=True, isVideo=False, isImage=False
    
    3. If specialists found:
       → Use default_model if in specialists
       → Else use first specialist
    
    4. If no specialists, find GENERALIST models:
       - Image: isImage=True (any)
       - Video: isVideo=True (any)
       - Audio: isAudio=True (any)
    
    5. If generalists found:
       → Use default_model if in generalists
       → Else use first generalist
    
    6. If no models found:
       → 422 Unprocessable Entity error
```

**Example Auto-Selection:**

```python
# Scenario: Upload video.mp4, no model specified
# Active models:
#   - SIGLIP-LSTM-V4: isVideo=True, isImage=False
#   - MFF-MoE: isVideo=True, isImage=True (multimodal)
#   - DistilDIRE: isImage=True, isVideo=False

# Step 1: Detect media type → VIDEO
# Step 2: Find specialists (isVideo=True ONLY)
#   → SIGLIP-LSTM-V4 ✅ (specialist)
#   → MFF-MoE ❌ (generalist, also does images)
# Step 3: Use SIGLIP-LSTM-V4 (specialist)
```

**Response (Success):**

```json
{
  "success": true,
  "model_used": "SIGLIP-LSTM-V4",
  "timestamp": "2025-10-26T10:30:45.123456",
  "data": {
    "prediction": "FAKE",
    "confidence": 0.8734,
    "processing_time": 12.456,
    "media_type": "video",
    "frame_count": 300,
    "frames_analyzed": 293,
    "frame_predictions": [
      {
        "index": 0,
        "score": 0.8234,
        "prediction": "FAKE"
      },
      {
        "index": 1,
        "score": 0.9123,
        "prediction": "FAKE"
      }
    ],
    "metrics": {
      "mean_score": 0.8734,
      "std_score": 0.0456,
      "max_score": 0.9789,
      "min_score": 0.7123,
      "fake_ratio": 0.96
    },
    "visualization_path": "http://localhost:3000/media/visualizations/abc123.mp4",
    "note": null
  }
}
```

**Response Schema:**

```python
class APIResponse(BaseModel, Generic[DataType]):
    """Standard wrapper for all successful API responses."""
    success: bool = True
    model_used: str                                    # Model that performed analysis
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: DataType                                     # VideoAnalysisResult or AudioAnalysisResult
```

**Video Analysis Result:**

```python
class VideoAnalysisResult(BaseAnalysisResult):
    prediction: str                      # "REAL" or "FAKE"
    confidence: float                    # 0.0 to 1.0
    processing_time: float               # seconds
    media_type: str = "video"
    frame_count: Optional[int]           # Total frames in video
    frames_analyzed: int                 # Frames/windows analyzed
    frame_predictions: List[FramePrediction]
    metrics: Union[
        SequenceBasedMetrics,            # For rolling window models
        FrameBasedMetrics,               # For per-frame models
        ColorCuesMetrics,                # For color-based models
        BlinkDetectionMetrics,           # For blink analysis
        FaceDetectionMetrics             # For face-based models
    ]
    visualization_path: Optional[str]    # URL to visualization video
    note: Optional[str]                  # Warnings, fallback info
```

**Audio Analysis Result:**

```python
class AudioAnalysisResult(BaseAnalysisResult):
    prediction: str                      # "REAL" or "FAKE"
    confidence: float                    # 0.0 to 1.0
    processing_time: float               # seconds
    media_type: str = "audio"
    properties: AudioProperties          # Duration, sample rate, channels
    pitch: PitchMetrics                  # Mean pitch, stability
    energy: EnergyMetrics                # RMS energy, silence ratio
    spectral: SpectralMetrics            # Spectral centroid, contrast
    voice_quality: Optional[VoiceQualityMetrics]
    visualization: Optional[AudioVisualization]  # Spectrogram URL
    note: Optional[str]
```

**Error Responses:**

```json
// 400 Bad Request - Empty file
{
  "error": "Bad Request",
  "message": "Uploaded media file is empty."
}

// 401 Unauthorized - Missing/invalid API key
{
  "error": "Unauthorized",
  "message": "Invalid or missing API Key"
}

// 415 Unsupported Media Type - Unknown file type
{
  "error": "Unsupported Media Type",
  "message": "Could not determine media type for file 'unknown.xyz'."
}

// 422 Unprocessable Entity - Invalid media
{
  "error": "Media Processing Error",
  "message": "Media could not be processed. Reason: Corrupted video file",
  "correlation_id": "abc123-req-uuid"
}

// 422 Unprocessable Entity - Model not found
{
  "error": "Validation Error",
  "message": "Requested model 'INVALID-MODEL' is not active or available."
}

// 422 Unprocessable Entity - No compatible models
{
  "error": "Validation Error",
  "message": "No active models found that can process media of type 'IMAGE'."
}

// 500 Internal Server Error - Inference failed
{
  "error": "Inference Error",
  "message": "Model inference failed and all fallbacks exhausted.",
  "primary_error": "CUDA out of memory",
  "fallback_error": "All fallback models also failed",
  "correlation_id": "abc123-req-uuid"
}
```

**Graceful Degradation (Fallback Models):**

```python
# If primary model fails, system attempts fallback models:

1. Primary model (e.g., SIGLIP-LSTM-V4) throws InferenceError
2. System finds other models supporting same media type
3. Tries each fallback in order:
   - SIGLIP-LSTM-V3
   - ColorCues-LSTM
   - EyeBlink-CNN-LSTM
4. If any fallback succeeds:
   → Returns result with note: "Primary model 'SIGLIP-LSTM-V4' failed. Used fallback model 'SIGLIP-LSTM-V3'."
5. If all fallbacks fail:
   → Returns 500 error with all error details
```

**CURL Examples:**

```bash
# Basic analysis (auto-select model)
curl -X POST http://localhost:8000/analyze \
  -H "X-API-Key: your_secret_key" \
  -F "media=@video.mp4"

# Specify model explicitly
curl -X POST http://localhost:8000/analyze \
  -H "X-API-Key: your_secret_key" \
  -F "media=@video.mp4" \
  -F "model=SIGLIP-LSTM-V4"

# With event publishing (for real-time progress)
curl -X POST http://localhost:8000/analyze \
  -H "X-API-Key: your_secret_key" \
  -F "media=@video.mp4" \
  -F "model=SIGLIP-LSTM-V4" \
  -F "mediaId=abc123-uuid" \
  -F "userId=user456-uuid"

# Analyze audio
curl -X POST http://localhost:8000/analyze \
  -H "X-API-Key: your_secret_key" \
  -F "media=@audio.mp3" \
  -F "model=MEL-Spectrogram-CNN-V2"

# Analyze image
curl -X POST http://localhost:8000/analyze \
  -H "X-API-Key: your_secret_key" \
  -F "media=@image.jpg" \
  -F "model=DistilDIRE-V1"
```

**JavaScript/Fetch Example:**

```javascript
// Frontend usage with FormData
const formData = new FormData();
formData.append('media', fileInput.files[0]);
formData.append('model', 'SIGLIP-LSTM-V4');
formData.append('mediaId', 'abc123-uuid');
formData.append('userId', 'user456-uuid');

const response = await fetch('http://localhost:8000/analyze', {
  method: 'POST',
  headers: {
    'X-API-Key': 'your_secret_key'
  },
  body: formData
});

const result = await response.json();
console.log(result.data.prediction);  // "FAKE" or "REAL"
console.log(result.data.confidence);  // 0.8734
```

---

## Status & Monitoring Endpoints

### GET / (Health Check)

**Public endpoint for service health monitoring.**

**Authentication:** None (public)

**Response:**

```json
{
  "status": "ok",
  "active_models": [
    {
      "name": "SIGLIP-LSTM-V4",
      "loaded": true,
      "description": "SIGLIP-LSTM-V4"
    },
    {
      "name": "MEL-Spectrogram-CNN-V2",
      "loaded": true,
      "description": "MEL-Spectrogram-CNN-V2"
    }
  ],
  "default_model": "SIGLIP-LSTM-V4"
}
```

**Use Case:** Load balancers, Kubernetes liveness probes

```yaml
# Kubernetes liveness probe
livenessProbe:
  httpGet:
    path: /
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
```

### GET /ping

**Simple ping endpoint.**

**Authentication:** None (public)

**Response:**

```json
{
  "status": "pong"
}
```

**Use Case:** Basic connectivity check

### GET /stats

**Comprehensive server statistics.**

**Authentication:** Required (X-API-Key)

**Response:**

```json
{
  "service_name": "Drishtiksha ML Inference Server",
  "version": "3.0.0",
  "status": "running",
  "uptime_seconds": 3600.45,
  "device_info": {
    "type": "cuda",
    "name": "NVIDIA GeForce RTX 3090",
    "total_memory": 24576.0,
    "used_memory": 18432.5,
    "free_memory": 6143.5,
    "memory_usage_percent": 75.0,
    "compute_capability": "8.6",
    "cuda_version": "11.8"
  },
  "system_info": {
    "python_version": "3.11.5",
    "platform": "Linux-5.15.0-generic-x86_64",
    "cpu_count": 16,
    "total_ram": 65536.0,
    "used_ram": 32768.0,
    "ram_usage_percent": 50.0,
    "uptime_seconds": 3600.45
  },
  "models_info": [
    {
      "name": "SIGLIP-LSTM-V4",
      "class_name": "SiglipLSTMV4",
      "description": "SIGLIP-LSTM-V4",
      "loaded": true,
      "device": "cuda",
      "model_path": "/app/models/SigLip-LSTM-v4.pth",
      "isAudio": false,
      "isVideo": true,
      "isImage": false,
      "isMultiModal": false,
      "memory_usage_mb": 2345.67,
      "load_time": 5.23,
      "inference_count": 42
    }
  ],
  "active_models_count": 3,
  "total_models_count": 3,
  "configuration": {
    "default_model": "SIGLIP-LSTM-V4",
    "active_models": ["SIGLIP-LSTM-V4", "MEL-Spectrogram-CNN-V2", "DistilDIRE-V1"],
    "device": "cuda",
    "redis_url": "redis://redis:6379",
    "storage_path": "/app/storage"
  }
}
```

**Use Case:** System monitoring, performance dashboards

### GET /device

**Detailed GPU/CPU information.**

**Authentication:** Required (X-API-Key)

**Response (CUDA):**

```json
{
  "type": "cuda",
  "name": "NVIDIA GeForce RTX 3090",
  "total_memory": 24576.0,
  "used_memory": 18432.5,
  "free_memory": 6143.5,
  "memory_usage_percent": 75.0,
  "compute_capability": "8.6",
  "cuda_version": "11.8"
}
```

**Response (CPU):**

```json
{
  "type": "cpu",
  "name": "Intel Core i9-12900K",
  "total_memory": null,
  "used_memory": null,
  "free_memory": null,
  "memory_usage_percent": null,
  "compute_capability": null,
  "cuda_version": null
}
```

### GET /system

**System resource information.**

**Authentication:** Required (X-API-Key)

**Response:**

```json
{
  "python_version": "3.11.5",
  "platform": "Linux-5.15.0-generic-x86_64",
  "cpu_count": 16,
  "total_ram": 65536.0,
  "used_ram": 32768.0,
  "ram_usage_percent": 50.0,
  "uptime_seconds": 3600.45
}
```

### GET /models

**Detailed model information.**

**Authentication:** Required (X-API-Key)

**Response:**

```json
{
  "models": [
    {
      "name": "SIGLIP-LSTM-V4",
      "class_name": "SiglipLSTMV4",
      "description": "SIGLIP-LSTM-V4",
      "loaded": true,
      "device": "cuda",
      "model_path": "/app/models/SigLip-LSTM-v4.pth",
      "isAudio": false,
      "isVideo": true,
      "isImage": false,
      "isMultiModal": false,
      "memory_usage_mb": 2345.67,
      "load_time": 5.23,
      "inference_count": 42
    }
  ],
  "summary": {
    "total_configured": 3,
    "currently_loaded": 3,
    "active_models": ["SIGLIP-LSTM-V4", "MEL-Spectrogram-CNN-V2", "DistilDIRE-V1"],
    "loaded_models": ["SIGLIP-LSTM-V4", "MEL-Spectrogram-CNN-V2", "DistilDIRE-V1"]
  }
}
```

### GET /config

**Server configuration summary.**

**Authentication:** Required (X-API-Key)

**Response:**

```json
{
  "default_model": "SIGLIP-LSTM-V4",
  "active_models": ["SIGLIP-LSTM-V4", "MEL-Spectrogram-CNN-V2", "DistilDIRE-V1"],
  "device": "cuda",
  "redis_url": "redis://redis:6379",
  "storage_path": "/app/storage",
  "assets_base_url": "http://localhost:3000"
}
```

### GET /health/deep

**Comprehensive model health check.**

**Authentication:** Required (X-API-Key)

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `force_refresh` | Boolean | `false` | Bypass cache and perform fresh checks |

**Response:**

```json
{
  "overall_status": "healthy",
  "timestamp": "2025-10-26T10:30:45.123456",
  "correlation_id": "abc123-req-uuid",
  "models": {
    "SIGLIP-LSTM-V4": {
      "status": "healthy",
      "load_status": "success",
      "inference_status": "success",
      "inference_time_ms": 234.56,
      "last_check": "2025-10-26T10:30:45.123456",
      "error": null
    },
    "MEL-Spectrogram-CNN-V2": {
      "status": "healthy",
      "load_status": "success",
      "inference_status": "success",
      "inference_time_ms": 156.78,
      "last_check": "2025-10-26T10:30:45.123456",
      "error": null
    }
  },
  "summary": {
    "total_models": 2,
    "healthy_models": 2,
    "unhealthy_models": 0,
    "health_percentage": 100.0
  }
}
```

**Use Case:** Pre-deployment validation, continuous monitoring

```bash
# Check health before deployment
curl -H "X-API-Key: your_key" \
  "http://localhost:8000/health/deep?force_refresh=true"

# Health check passed (200 OK)
# → Proceed with deployment

# Health check failed (500 Internal Server Error)
# → Rollback deployment
```

### POST /health/clear-cache

**Clear health check cache.**

**Authentication:** Required (X-API-Key)

**Response:**

```json
{
  "status": "success",
  "message": "Health check cache cleared"
}
```

---

## Middleware

### Correlation ID Middleware

**Automatic request tracing for distributed systems.**

**How it works:**

1. Client sends request (optionally with `X-Correlation-ID` header)
2. Middleware checks for existing correlation ID
3. If missing, generates new UUID
4. Sets correlation ID in request context
5. Logs request start with correlation ID
6. Processes request
7. Adds correlation ID to response headers
8. Logs request completion with correlation ID

**Implementation:**

```python
@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    # Check if client provided a correlation ID
    correlation_id = request.headers.get("X-Correlation-ID")
    
    # Generate one if not provided
    if not correlation_id:
        correlation_id = generate_correlation_id()
    
    # Set in context for the request
    set_correlation_id(correlation_id)
    
    # Log the request start
    logger.info(
        f"[{correlation_id}] {request.method} {request.url.path} - Started"
    )
    
    try:
        response = await call_next(request)
        
        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id
        
        # Log the request completion
        logger.info(
            f"[{correlation_id}] {request.method} {request.url.path} - "
            f"Completed {response.status_code}"
        )
        
        return response
    except Exception as e:
        logger.error(
            f"[{correlation_id}] {request.method} {request.url.path} - "
            f"Failed with exception: {e}",
            exc_info=True
        )
        raise
```

**Example Logs:**

```text
10:30:45 | INFO     | src.app.main            | [abc123-uuid] POST /analyze - Started
10:30:45 | INFO     | src.ml.registry         | [abc123-uuid] Loading model: SIGLIP-LSTM-V4
10:30:57 | INFO     | src.app.routers.analysis| [abc123-uuid] Analysis complete: FAKE (0.8734)
10:30:57 | INFO     | src.app.main            | [abc123-uuid] POST /analyze - Completed 200
```

**Benefits:**

- **Distributed Tracing**: Track requests across microservices
- **Debugging**: Find all logs for a specific request
- **Client-Side Tracking**: Frontend can pass correlation ID for end-to-end tracing

**Client Usage:**

```javascript
// Frontend generates correlation ID
const correlationId = crypto.randomUUID();

const response = await fetch('http://localhost:8000/analyze', {
  method: 'POST',
  headers: {
    'X-API-Key': 'your_key',
    'X-Correlation-ID': correlationId  // Pass correlation ID
  },
  body: formData
});

// Server echoes it back in response
console.log(response.headers.get('X-Correlation-ID'));  // Same ID
```

---

## Exception Handlers

### Global Exception Handler

**Catches all unhandled exceptions.**

```python
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    correlation_id = get_correlation_id() or "unknown"
    logger.critical(
        f"[{correlation_id}] Unhandled exception during request to {request.url}: {exc}",
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content=APIError(
            error="Internal Server Error",
            message="An unexpected error occurred. The technical team has been notified.",
            details={"correlation_id": correlation_id}
        ).model_dump(),
    )
```

**Example:**

```python
# Unhandled exception in route
@router.get("/crash")
def crash_endpoint():
    raise RuntimeError("Something went wrong!")

# Response (500 Internal Server Error):
{
  "error": "Internal Server Error",
  "message": "An unexpected error occurred. The technical team has been notified.",
  "details": {
    "correlation_id": "abc123-uuid"
  }
}
```

### ValueError Exception Handler

**Handles validation errors.**

```python
@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    correlation_id = get_correlation_id() or "unknown"
    logger.warning(
        f"[{correlation_id}] Validation error for request {request.url}: {exc}",
        exc_info=False
    )
    return JSONResponse(
        status_code=422,
        content=APIError(
            error="Validation Error",
            message=str(exc),
            details={"correlation_id": correlation_id}
        ).model_dump(),
    )
```

### NotImplementedError Handler

**Handles unimplemented features.**

```python
@app.exception_handler(NotImplementedError)
async def not_implemented_error_handler(request: Request, exc: NotImplementedError):
    correlation_id = get_correlation_id() or "unknown"
    logger.warning(
        f"[{correlation_id}] Feature not implemented for request {request.url}: {exc}",
        exc_info=False
    )
    return JSONResponse(
        status_code=501,
        content=APIError(
            error="Not Implemented",
            message=str(exc),
            details={"correlation_id": correlation_id}
        ).model_dump(),
    )
```

---

## Dependencies

### ModelManager Dependency

**Provides singleton ModelManager instance.**

```python
app_state: Dict[str, Any] = {}

def get_model_manager() -> ModelManager:
    """Dependency to get the singleton ModelManager instance."""
    manager = app_state.get("model_manager")
    if not manager:
        raise HTTPException(
            status_code=503,
            detail="ModelManager is not available. The service may be starting up or in a failed state."
        )
    return manager

# Usage in route:
@router.post("/analyze")
async def analyze_media(
    model_manager: ModelManager = Depends(get_model_manager)
):
    model = model_manager.get_model("SIGLIP-LSTM-V4")
    ...
```

### Media Processing Dependency

**Handles file upload, temporary storage, and model selection.**

```python
async def process_media_request(
    media: UploadFile = Form(..., alias="media"),
    model_name_form: str = Form(default=None, alias="model"),
    model_manager: ModelManager = Depends(get_model_manager)
) -> AsyncGenerator[Tuple[str, str], None]:
    """
    Processes uploaded media file and selects appropriate model.
    
    Yields:
        Tuple[str, str]: (model_name, temp_file_path)
    """
    temp_path = None
    try:
        # Step 1: Save to temporary file
        suffix = os.path.splitext(media.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name

        async with aiofiles.open(temp_path, 'wb') as out_file:
            while content := await media.read(1024 * 1024):  # 1MB chunks
                await out_file.write(content)

        # Step 2: Determine media type
        media_type = get_media_type(temp_path)  # VIDEO, AUDIO, IMAGE, UNKNOWN
        
        # Step 3: Select model (auto or explicit)
        if model_name_form:
            # Explicit model selection
            target_model_name = model_name_form
        else:
            # Auto-select based on media type
            target_model_name = auto_select_model(media_type, model_manager)
        
        # Step 4: Validate compatibility
        config = model_manager.get_active_model_configs()[target_model_name]
        if not is_compatible(media_type, config):
            raise HTTPException(422, "Model doesn't support this media type")
        
        yield target_model_name, temp_path

    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
```

**Auto-Selection Algorithm:**

```python
def auto_select_model(media_type: str, manager: ModelManager) -> str:
    configs = manager.get_active_model_configs()
    
    # 1. Find specialist models (single media type)
    specialists = [
        name for name, config in configs.items()
        if is_specialist(config, media_type)
    ]
    
    if specialists:
        # Prefer default model if it's a specialist
        if settings.default_model_name in specialists:
            return settings.default_model_name
        return specialists[0]
    
    # 2. Find generalist models (supports media type)
    generalists = [
        name for name, config in configs.items()
        if supports_media_type(config, media_type)
    ]
    
    if generalists:
        if settings.default_model_name in generalists:
            return settings.default_model_name
        return generalists[0]
    
    # 3. No compatible models
    raise HTTPException(422, f"No models support {media_type}")

def is_specialist(config, media_type: str) -> bool:
    """Check if model is a specialist (supports ONLY this media type)."""
    if media_type == "VIDEO":
        return config.isVideo and not config.isAudio and not config.isImage
    elif media_type == "AUDIO":
        return config.isAudio and not config.isVideo and not config.isImage
    elif media_type == "IMAGE":
        return config.isImage and not config.isVideo and not config.isAudio
    return False
```

---

## Application Lifecycle

### Startup (lifespan)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application's lifespan."""
    logger.info(f"🔼 Starting up {settings.project_name}...")
    
    # 1. Initialize ModelManager
    manager = ModelManager(settings)
    app_state["model_manager"] = manager

    # 2. Eagerly load all active models
    try:
        manager.load_models()  # Loads all ACTIVE_MODELS
    except ModelRegistryError as e:
        logger.critical(f"❌ FATAL: Critical model failed to load. Reason: {e}")
        sys.exit(1)  # Exit with error code for orchestrators

    logger.info(f"✅ {settings.project_name} startup complete.")
    yield  # Application runs here
    
    # Shutdown
    logger.info(f"🔽 Shutting down {settings.project_name}.")
    app_state.clear()
```

**Startup Flow:**

```text
1. Load configuration (config.yaml + .env)
2. Initialize FastAPI app
3. Run lifespan startup:
   a. Create ModelManager
   b. Load all ACTIVE_MODELS
   c. If any model fails → exit(1)
4. Register routers
5. Register middleware
6. Start Uvicorn server
7. Begin accepting requests
```

**Graceful Shutdown:**

```text
1. Receive SIGTERM signal
2. Run lifespan shutdown:
   a. Clear app_state
   b. Release resources
3. Stop Uvicorn server
4. Exit cleanly
```

---

## OpenAPI Documentation

**FastAPI automatically generates interactive API docs:**

### Swagger UI

**URL:** `http://localhost:8000/docs`

**Features:**

- Interactive API testing
- Request/response schemas
- Authentication testing (X-API-Key input)
- Example requests/responses

### ReDoc

**URL:** `http://localhost:8000/redoc`

**Features:**

- Clean, three-column layout
- Comprehensive schema documentation
- Markdown support
- Better for reading than testing

### OpenAPI JSON

**URL:** `http://localhost:8000/openapi.json`

**Use Case:** Import into API testing tools (Postman, Insomnia)

---

## Error Response Format

**All errors follow consistent schema:**

```python
class APIError(BaseModel):
    """Standard model for API error responses."""
    error: str                                # Error category
    message: str                              # Human-readable message
    details: Optional[Union[str, Dict[str, Any]]] = None  # Additional context
```

**HTTP Status Codes:**

| Code | Name | When Used |
|------|------|-----------|
| 200 | OK | Successful analysis |
| 400 | Bad Request | Empty file, malformed request |
| 401 | Unauthorized | Missing/invalid API key |
| 415 | Unsupported Media Type | Unknown file type |
| 422 | Unprocessable Entity | Invalid media, model incompatibility |
| 500 | Internal Server Error | Unexpected errors, inference failures |
| 501 | Not Implemented | Unimplemented features |
| 503 | Service Unavailable | Server starting up, ModelManager unavailable |

---

## Best Practices

### 1. Always Include Correlation ID

```javascript
// Frontend should generate and track correlation IDs
const correlationId = crypto.randomUUID();

fetch('/analyze', {
  headers: {
    'X-Correlation-ID': correlationId
  }
});

// On error, show correlation ID to user
alert(`Error processing request. Correlation ID: ${correlationId}`);
// Support team can search logs by this ID
```

### 2. Handle Fallback Gracefully

```javascript
// Check for fallback usage in response
if (result.data.note?.includes('fallback')) {
  console.warn('Primary model failed, used fallback');
  // Show warning to user
}
```

### 3. Validate Before Upload

```javascript
// Frontend validation
const file = fileInput.files[0];

// Check file size (e.g., 100MB limit)
if (file.size > 100 * 1024 * 1024) {
  alert('File too large (max 100MB)');
  return;
}

// Check file type
const allowedTypes = ['video/mp4', 'video/avi', 'audio/mp3', 'image/jpeg'];
if (!allowedTypes.includes(file.type)) {
  alert('Unsupported file type');
  return;
}
```

### 4. Use Appropriate Models

```javascript
// Prefer specialist models for best accuracy
const modelsByMediaType = {
  video: 'SIGLIP-LSTM-V4',        // Video specialist
  audio: 'MEL-Spectrogram-CNN-V2', // Audio specialist
  image: 'DistilDIRE-V1'          // Image specialist
};

const model = modelsByMediaType[mediaType];
formData.append('model', model);
```

---

## Summary

The FastAPI REST API provides:

✅ **Comprehensive Analysis** - `/analyze` endpoint with auto model selection  
✅ **Robust Authentication** - API key with timing attack prevention  
✅ **Graceful Degradation** - Automatic fallback on model failures  
✅ **Distributed Tracing** - Correlation ID middleware for request tracking  
✅ **Health Monitoring** - Multiple endpoints for service health checks  
✅ **Error Handling** - Consistent error responses with correlation IDs  
✅ **Auto Documentation** - Swagger UI and ReDoc  
✅ **Type Safety** - Pydantic schemas for all requests/responses  

**Key Integration Points:**

1. **Frontend** → REST API → **ModelManager** → Detectors
2. **Load Balancers** → `/` health check → Service routing
3. **Monitoring Systems** → `/stats`, `/health/deep` → Dashboards
4. **Client Apps** → Correlation IDs → Distributed tracing

**Production Deployment:**

```bash
# Environment variables
API_KEY=your_secret_key_here
DEFAULT_MODEL_NAME=SIGLIP-LSTM-V4
ACTIVE_MODELS=["SIGLIP-LSTM-V4","MEL-Spectrogram-CNN-V2","DistilDIRE-V1"]
DEVICE=cuda
STORAGE_PATH=/app/storage
ASSETS_BASE_URL=https://api.example.com

# Run with Uvicorn
uvicorn src.app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info
```
