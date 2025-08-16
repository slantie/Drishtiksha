# Drishtiksha: Deepfake Detection Service v2.0

A high-performance, production-ready AI service for deepfake detection. This fully-refactored application features a modular architecture, multiple state-of-the-art models, comprehensive API endpoints, and real-time server monitoring capabilities. Built with FastAPI and PyTorch, this service provides a robust and scalable solution designed for mission-critical environments.

## 🚀 Key Features

### Core Capabilities

-   **Multi-Model Architecture:** Deploy multiple detection models simultaneously, with modular activation through environment configuration for optimal resource management.
-   **Production-Ready Performance:** Built with **FastAPI** and Uvicorn for high-throughput, asynchronous request handling with CUDA acceleration.
-   **Comprehensive Analysis API:** Four distinct analysis endpoints (`/analyze`, `/analyze/detailed`, `/analyze/frames`, `/analyze/visualize`) catering to different use cases, from quick checks to forensic analysis.
-   **Real-time Monitoring:** Extensive server statistics endpoints for monitoring device usage, system resources, model status, and performance metrics.
-   **Dynamic Model Management:** Environment-controlled model activation with automatic filtering and health monitoring.

### Architectural Highlights (Post-Refactor)

-   **Modular API Layer:** FastAPI endpoints logically separated into routers (`status`, `analysis`) with clean dependency injection patterns.
-   **Decoupled ML Layer:** Clear separation between model architecture, inference logic, and model management registry.
-   **Type-Safe Configuration:** Centralized Pydantic `Settings` with strict validation from `.env` and `config.yaml`.
-   **Comprehensive Monitoring:** Real-time system statistics, device monitoring, and model performance tracking.
-   **Environment-Based Control:** Dynamic model activation through `ACTIVE_MODELS` environment variable.

## 🤖 Available Models

### 1\. SIGLIP-LSTM V3 (`SIGLIP-LSTM-V3`)

-   **Architecture:** An enhanced SIGLIP vision encoder combined with an LSTM for advanced temporal feature analysis.
-   **Strengths:** The most accurate model in the suite, providing detailed, frame-by-frame analysis with temporal smoothing.
-   **Features:** Supports all four analysis endpoints, including detailed metrics and video visualization.
-   **Use Cases:** High-stakes forensic analysis, detailed reporting, and visual evidence generation.

### 2\. SIGLIP-LSTM V1 (`SIGLIP-LSTM-V1`)

-   **Architecture:** A foundational SIGLIP vision encoder with a standard LSTM for temporal analysis.
-   **Strengths:** Offers a strong balance of speed and accuracy.
-   **Features:** Supports the `/analyze` (quick) endpoint only.
-   **Use Cases:** General-purpose deepfake screening and real-time applications where speed is a priority.

### 3\. ColorCues LSTM (`COLOR-CUES-LSTM-V1`)

-   **Architecture:** Utilizes R-G color histograms from dlib-detected facial landmarks, fed into an LSTM.
-   **Strengths:** Specialized in detecting deepfakes created through color manipulation or that exhibit subtle color artifacts.
-   **Features:** Supports all four analysis endpoints.
-   **Use Cases:** Detecting color-based manipulation and analyzing videos with a strong focus on facial regions.

## 📋 API Endpoints

All analysis endpoints are secured and require an API key to be sent in the `X-API-Key` header.

### Status & Monitoring Endpoints

-   `GET /`: Provides a detailed health check of the service, including the status of all configured models.
-   `GET /ping`: A simple endpoint to confirm the server is running and responsive.
-   `GET /stats`: **[NEW]** Comprehensive server statistics including device info, system resources, model details, and configuration.
-   `GET /device`: **[NEW]** Detailed GPU/CPU device information including memory usage, compute capability, and CUDA status.
-   `GET /system`: **[NEW]** System resource information including RAM usage, CPU count, platform details, and uptime.
-   `GET /models`: **[NEW]** Detailed information about all configured models, their status, and memory usage.
-   `GET /config`: **[NEW]** Server configuration summary including active models, device settings, and version information.

### Analysis Endpoints

All analysis endpoints accept multipart/form-data with two fields: `video` (the video file) and an optional `model` (the string name of the model to use, e.g., `SIGLIP-LSTM-V3`).

#### 1\. Quick Analysis

Performs a fast, high-level analysis.

```http
POST /analyze
```

-   **Model Support:** All models.
-   **Response:** `APIResponse[QuickAnalysisData]` containing the prediction, confidence score, processing time, and any relevant notes.

#### 2\. Detailed Analysis

Provides a comprehensive analysis with model-specific metrics.

```http
POST /analyze/detailed
```

-   **Model Support:** `SIGLIP-LSTM-V3`, `COLOR-CUES-LSTM-V1`.
-   **Response:** `APIResponse[DetailedAnalysisData]` containing the prediction, confidence, processing time, and a dictionary of detailed metrics.

#### 3\. Frame-by-Frame Analysis

Returns a prediction for each frame or sequence in the video.

```http
POST /analyze/frames
```

-   **Model Support:** `SIGLIP-LSTM-V3`, `COLOR-CUES-LSTM-V1`.
-   **Response:** `APIResponse[FramesAnalysisData]` containing an overall prediction and a list of per-frame/sequence predictions.

#### 4\. Visual Analysis

Generates and returns a new video file with an overlaid analysis graph.

```http
POST /analyze/visualize
```

-   **Model Support:** `SIGLIP-LSTM-V3`, `COLOR-CUES-LSTM-V1`.
-   **Response:** A `video/mp4` file stream.

## 🏗️ Project Structure (Refactored)

```text
/Server
├── .env              # Environment variables (API Key, default model)
├── pyproject.toml    # Python dependencies (managed by uv)
├── uv.lock           # Lock file for reproducible builds
├── README.md         # This documentation
├── test_e2e_api.py   # End-to-end test suite for all endpoints
│
├── configs/
│   └── config.yaml   # Centralized model configurations
│
├── models/           # Pre-trained model weights
│
├── scripts/
│   └── predict.py    # Standalone CLI prediction script
│
├── assets/
│   └── id0_0001.mp4  # Sample test video
│
└── src/
    ├── config.py     # Pydantic settings management
    │
    ├── app/          # FastAPI application layer
    │   ├── main.py   # App entrypoint, lifespan manager, routers
    │   ├── schemas.py# Pydantic models for API requests/responses
    │   ├── security.py # API key authentication
    │   ├── dependencies.py # Shared dependencies for request processing
    │   └── routers/  # API endpoint definitions
    │       ├── analysis.py
    │       └── status.py
    │
    └── ml/             # Machine learning inference layer
        ├── base.py     # Abstract base class for all models
        ├── registry.py # Model factory and manager
        ├── utils.py    # ML utilities (frame extraction)
        ├── architectures/ # PyTorch model architecture definitions
        │   ├── siglip_lstm.py
        │   └── color_cues_lstm.py
        └── models/     # Model inference logic implementations
            ├── siglip_lstm_detector.py
            └── color_cues_detector.py
```

## 🚀 Setup and Installation

### Prerequisites

-   Python 3.10+
-   CUDA 11.8+ (Strongly recommended for GPU acceleration)
-   `uv` (The project's package manager)
-   FFmpeg

### Installation

1.  **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    cd Server
    ```
2.  **Create and Activate Virtual Environment**
    ```bash
    uv venv
    source .venv/bin/activate  # On Linux/Mac
    .venv\Scripts\activate    # On Windows
    ```
3.  **Install Dependencies**
    ```bash
    uv sync
    ```
4.  **Download Models**
    Ensure the required `.pth` and `.dat` files are placed in the `/models` directory.
5.  **Configure Environment**
    Create a `.env` file in the project root by copying `.env.example` and configure the service settings.

    ```env
    # Security
    API_KEY="your_super_secret_api_key_here_at_least_32_characters"

    # Model Configuration
    DEFAULT_MODEL_NAME="SIGLIP-LSTM-V3"

    # Device Configuration - Choose compute device for model inference
    # Options: "cuda" (GPU acceleration), "cpu" (CPU-only processing)
    DEVICE="cuda"

    # Active Models - Control which models are loaded and served
    # Available: SIGLIP-LSTM-V1, SIGLIP-LSTM-V3, COLOR-CUES-LSTM-V1
    ACTIVE_MODELS="SIGLIP-LSTM-V3,COLOR-CUES-LSTM-V1"
    ```

## 🏃‍♂️ Running the Application

### Environment-Based Configuration

The service supports dynamic configuration through environment variables:

#### Model Control

-   **Production Setup**: `ACTIVE_MODELS="SIGLIP-LSTM-V3,COLOR-CUES-LSTM-V1"`
-   **Development**: `ACTIVE_MODELS="SIGLIP-LSTM-V1"`
-   **Full Suite**: `ACTIVE_MODELS="SIGLIP-LSTM-V1,SIGLIP-LSTM-V3,COLOR-CUES-LSTM-V1"`

#### Device Configuration

-   **GPU Acceleration**: `DEVICE="cuda"` (Recommended for production)
-   **CPU Only**: `DEVICE="cpu"` (Fallback or CPU-only environments)

Only the models listed in `ACTIVE_MODELS` will be loaded into memory, and all models will use the specified `DEVICE`, allowing for:

-   **Faster startup times** (fewer models to load)
-   **Lower memory usage** (inactive models are ignored)
-   **Environment-specific deployments** (different models and devices per environment)
-   **Flexible hardware targeting** (GPU for production, CPU for development)

### Development Server

For local development with hot-reloading.

```bash
uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Server

For deployment. It is recommended to use a process manager like Gunicorn with Uvicorn workers for production.

```bash
uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

Upon starting, the server will pre-load all configured models into memory and will then be ready to accept requests.

## 🧪 Testing the Service

With the server running, you can execute the end-to-end test suite from a separate terminal.

1.  Ensure your `API_KEY` in `test_e2e_api.py` matches your `.env` file.
2.  Run the test:
    ```bash
    python test_e2e_api.py
    ```

### Interactive Documentation

Once the server is running, you can access the interactive API documentation (Swagger UI) in your browser:

-   **URL:** `http://localhost:8000/docs`

## 📊 Server Monitoring & Statistics

The service provides comprehensive monitoring endpoints for production deployments:

### Real-time Statistics (`GET /stats`)

```json
{
    "service_name": "Drishtiksha: Deepfake Detection",
    "version": "2.0.0",
    "status": "running",
    "uptime_seconds": 3600.45,
    "device_info": {
        "type": "cuda",
        "name": "NVIDIA GeForce RTX 4090",
        "total_memory": 24.0,
        "used_memory": 8.5,
        "free_memory": 15.5,
        "memory_usage_percent": 35.4,
        "compute_capability": "8.9",
        "cuda_version": "12.1"
    },
    "system_info": {
        "python_version": "3.11.0",
        "platform": "Linux-5.15.0-ubuntu",
        "cpu_count": 16,
        "total_ram": 32.0,
        "used_ram": 12.8,
        "ram_usage_percent": 40.0,
        "uptime_seconds": 3600.45
    },
    "active_models_count": 2,
    "total_models_count": 3,
    "configuration": {
        "active_models": ["SIGLIP-LSTM-V3", "COLOR-CUES-LSTM-V1"],
        "default_model": "SIGLIP-LSTM-V3",
        "cuda_available": true
    }
}
```

### Device Monitoring (`GET /device`)

-   GPU/CPU status and specifications
-   Memory usage (total, used, free)
-   CUDA availability and version
-   Compute capability information

### System Resources (`GET /system`)

-   Python version and platform information
-   CPU count and RAM usage
-   System uptime and resource utilization

### Model Information (`GET /models`)

-   Detailed status of all configured models
-   Memory usage per model
-   Load status and configuration details
-   Active vs configured models summary

## 📦 Docker Deployment

A `Dockerfile` and `docker-compose.yml` are provided for easy containerization and deployment.

1.  **Build the Docker Image:**
    ```bash
    docker build -t deepfake-detection-service .
    ```
2.  **Run with Docker Compose:**
    (Ensure you have NVIDIA's container toolkit installed for GPU support)
    ```bash
    docker-compose up
    ```

This will build the service, mount the necessary volumes, and expose the application on port 8000.

---

**Built with ❤️ for advanced deepfake detection and video authenticity verification.**
