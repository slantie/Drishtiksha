# Drishtiksha: Deepfake Detection Service v2.0

A high-performance, production-ready AI service for deepfake detection. This fully-refactored application features a modular architecture, multiple state-of-the-art models, and a comprehensive API for video authenticity verification. Built with FastAPI and PyTorch, this service provides a robust and scalable solution designed for mission-critical environments.

## 🚀 Key Features

### Core Capabilities

-   **Multi-Model Architecture:** Deploy multiple detection models simultaneously, with all models pre-loaded into GPU memory for instant, low-latency inference.
-   **Production-Ready Performance:** Built with **FastAPI** and Uvicorn for high-throughput, asynchronous request handling.
-   **Comprehensive Analysis API:** Four distinct analysis endpoints (`/analyze`, `/analyze/detailed`, `/analyze/frames`, `/analyze/visualize`) catering to different use cases, from quick checks to forensic analysis.
-   **CUDA Acceleration:** Full GPU support for accelerated inference with graceful fallback to CPU if CUDA is unavailable.

### Architectural Highlights (Post-Refactor)

-   **Modular API Layer:** FastAPI endpoints are logically separated into routers (`status`, `analysis`) with a clean, dependency-injection pattern for handling requests, which has eliminated all repetitive code.
-   **Decoupled ML Layer:** A clear separation between model architecture (`src/ml/architectures`), inference logic (`src/ml/models`), and the model management registry.
-   **Type-Safe Centralized Configuration:** A single Pydantic `Settings` object (`src/config.py`) loads and strictly validates all configurations from `.env` and `config.yaml`, providing a single source of truth for the application.
-   **Robust Error Handling:** The service now provides standardized, detailed JSON error responses for all failed requests. Models are designed to be resilient, offering graceful fallbacks for challenging video inputs.

## 🤖 Available Models

### 1\. SIGLIP-LSTM V3 (`siglip-lstm-v3`)

-   **Architecture:** An enhanced SIGLIP vision encoder combined with an LSTM for advanced temporal feature analysis.
-   **Strengths:** The most accurate model in the suite, providing detailed, frame-by-frame analysis with temporal smoothing.
-   **Features:** Supports all four analysis endpoints, including detailed metrics and video visualization.
-   **Use Cases:** High-stakes forensic analysis, detailed reporting, and visual evidence generation.

### 2\. SIGLIP-LSTM V1 (`siglip-lstm-v1`)

-   **Architecture:** A foundational SIGLIP vision encoder with a standard LSTM for temporal analysis.
-   **Strengths:** Offers a strong balance of speed and accuracy.
-   **Features:** Supports the `/analyze` (quick) endpoint only.
-   **Use Cases:** General-purpose deepfake screening and real-time applications where speed is a priority.

### 3\. ColorCues LSTM (`color-cues-lstm-v1`)

-   **Architecture:** Utilizes R-G color histograms from dlib-detected facial landmarks, fed into an LSTM.
-   **Strengths:** Specialized in detecting deepfakes created through color manipulation or that exhibit subtle color artifacts.
-   **Features:** Supports all four analysis endpoints.
-   **Use Cases:** Detecting color-based manipulation and analyzing videos with a strong focus on facial regions.

## 📋 API Endpoints

All analysis endpoints are secured and require an API key to be sent in the `X-API-Key` header.

### Status Endpoints

-   `GET /`: Provides a detailed health check of the service, including the status of all configured models.
-   `GET /ping`: A simple endpoint to confirm the server is running and responsive.

### Analysis Endpoints

All analysis endpoints accept multipart/form-data with two fields: `video` (the video file) and an optional `model` (the string name of the model to use, e.g., `siglip-lstm-v3`).

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

-   **Model Support:** `siglip-lstm-v3`, `color-cues-lstm-v1`.
-   **Response:** `APIResponse[DetailedAnalysisData]` containing the prediction, confidence, processing time, and a dictionary of detailed metrics.

#### 3\. Frame-by-Frame Analysis

Returns a prediction for each frame or sequence in the video.

```http
POST /analyze/frames
```

-   **Model Support:** `siglip-lstm-v3`, `color-cues-lstm-v1`.
-   **Response:** `APIResponse[FramesAnalysisData]` containing an overall prediction and a list of per-frame/sequence predictions.

#### 4\. Visual Analysis

Generates and returns a new video file with an overlaid analysis graph.

```http
POST /analyze/visualize
```

-   **Model Support:** `siglip-lstm-v3`, `color-cues-lstm-v1`.
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
    Create a `.env` file in the project root by copying `.env.example` and set your `API_KEY`.
    ```env
    API_KEY="your_super_secret_api_key_here_at_least_32_characters"
    DEFAULT_MODEL_NAME="siglip-lstm-v3"
    ```

## 🏃‍♂️ Running the Application

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
