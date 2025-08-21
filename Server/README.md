# Drishtiksha: Deepfake Detection Server v3.0

Drishtiksha is a high-performance, modular, and extensible deepfake detection server built with FastAPI. It provides a robust REST API for analyzing video and audio files using a suite of diverse, state-of-the-art deep learning models. The system is designed for scalability and detailed analysis, offering features like frame-by-frame inspection, visual overlays, and real-time progress reporting via Redis.

## Overview

This project provides a comprehensive backend solution for deepfake detection. Key features include:

*   **Multi-Model Architecture:** Dynamically load and serve multiple detection models, each targeting different deepfake artifacts (e.g., facial inconsistencies, eye-blinking patterns, color cues, audio anomalies).
*   **Comprehensive API:** Offers a range of endpoints from quick predictions to detailed, frame-by-frame analysis and visual feedback.
*   **High Performance:** Built on FastAPI and Uvicorn for asynchronous request handling, with ML model inference running in separate threads to prevent blocking.
*   **Rich Analytics:** Delivers detailed metrics, including confidence scores, per-frame suspicion levels, temporal consistency, and processing breakdowns.
*   **Dynamic Configuration:** Server behavior, active models, and device (CPU/GPU) are easily configured via `.env` and YAML files without changing the code.
*   **Real-time Feedback:** Integrates with Redis to publish progress events, allowing clients to monitor long-running analysis tasks.
*   **System Monitoring:** Includes endpoints for checking service health, resource usage (CPU/GPU), and the status of all configured models.

---

## Project Structure

The project is organized into a clean and logical structure, separating the application logic from the machine learning components.

```
zaptrixio-cyber-drishtiksha/
├── pyproject.txt             # Project dependencies
└── Server/
    ├── pyproject.toml        # Main project dependencies and metadata for `uv`
    ├── .env.example          # Template for environment variables
    ├── .python-version       # Specifies the required Python version (3.12)
    ├── main.py               # A simple entry point script
    ├── assets/               # For static assets (if any)
    ├── configs/
    │   └── config.yaml       # Core configuration for all models and settings
    ├── models/               # Directory to store trained model weights (.pth, .dat files)
    ├── scripts/
    │   ├── convert_weights.py # Utility to convert Keras models to PyTorch
    │   └── predict.py         # A CLI tool for running local predictions
    └── src/
        ├── app/              # Contains all FastAPI-related components
        │   ├── routers/      # API endpoint definitions (analysis, status)
        │   ├── dependencies.py # Reusable dependencies for API routes
        │   ├── main.py       # FastAPI application entry point and lifespan manager
        │   ├── schemas.py    # Pydantic models for API request/response validation
        │   └── security.py   # API key authentication logic
        └── ml/               # Contains all machine learning components
            ├── architectures/ # PyTorch model architecture definitions (NN modules)
            ├── models/       # High-level model handlers (loading, preprocessing, inference)
            ├── base.py       # Abstract BaseModel class defining the model interface
            ├── event_publisher.py # Handles publishing progress events to Redis
            ├── registry.py   # The ModelManager for loading and accessing models
            ├── system_info.py # Functions for gathering system/device statistics
            └── utils.py      # Utility functions (e.g., frame extraction)
```

---

## How to Setup

### Prerequisites

*   **Python 3.12**
*   **uv** (or `pip`) for package management. `uv` is recommended for speed.
*   **Redis Server:** Required for real-time progress event publishing.
*   **System Dependencies for dlib:** The `dlib` library requires `CMake` and a C++ compiler.
    *   On Debian/Ubuntu: `sudo apt-get update && sudo apt-get install build-essential cmake`
    *   On macOS: `brew install cmake`

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/zaptrixio-cyber/Drishtiksha.git
    cd zaptrixio-cyber-drishtiksha/Server/
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    # Using uv (recommended)
    uv venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    # Using uv sync (recommended)
    uv sync
    ```
    OR
    ```bash
    # Using uv
    uv pip install -e .
    ```

4.  **Configure Environment Variables**
    Copy the example `.env` file and customize it.
    ```bash
    cp .env.example .env
    ```
    Now, edit the `.env` file:
    *   `API_KEY`: Generate a secure, random 32-character string.
    *   `DEFAULT_MODEL_NAME`: Set the default model (e.g., "SIGLIP-LSTM-V3").
    *   `DEVICE`: Set to `cuda` for GPU acceleration or `cpu`.
    *   `REDIS_URL`: The connection URL for your Redis server.
    *   `ACTIVE_MODELS`: A comma-separated list of models from `config.yaml` to load on startup.

5.  **Download Model Weights**
    You must acquire the necessary model weight files (`.pth`, `.dat`, etc.) and place them in the `Server/models/` directory. The expected file paths for each model are defined in `configs/config.yaml`.

### Running the Server

Once the setup is complete, you can start the application using Uvicorn.

```bash
uv run uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

The server will start, pre-load all `ACTIVE_MODELS`, and be ready to accept requests on port 8000.

---

## Models

The server supports a variety of models, each with a unique approach to deepfake detection. Models are defined in `configs/config.yaml` and loaded by the `ModelManager`.

| Model Name             | Description & Methodology                                                                                                                                                             | Type    |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| **SIGLIP-LSTM-V1/V3/V4** | Uses a powerful SigLIP (Sigmoid Loss for Language Image Pre-Training) vision transformer to extract features from video frames, which are then fed into a Bi-LSTM to analyze temporal patterns. V3 and V4 offer more detailed analysis and improved architectures. | Video |
| **COLOR-CUES-LSTM-V1** | Detects deepfakes by analyzing color inconsistencies. It uses dlib to find facial landmarks, crops the face, calculates 2D (r,g) chromaticity histograms for each frame, and uses an LSTM to classify sequences of these histograms. | Video |
| **EFFICIENTNET-B7-V1** | A frame-by-frame detector. It uses an MTCNN to find faces in each frame and then classifies each face using a powerful EfficientNet-B7 model. Results are aggregated using a confidence-based strategy. | Video |
| **EYEBLINK-CNN-LSTM-V1** | Focuses on unnatural eye blinking patterns. It detects blinks by calculating the Eye Aspect Ratio (EAR). Sequences of cropped eye regions during blinks are then analyzed by a CNN (Xception) + LSTM architecture. | Video |
| **SCATTERING-WAVE-V1** | An audio-only model that detects audio deepfakes (e.g., voice cloning). It converts the audio track into a Mel Spectrogram and uses a 2D Wavelet Scattering Transform for feature extraction, followed by a classifier. | Audio   |

---

## System Architecture

The application is built around a modular and decoupled architecture, ensuring maintainability and scalability.

1.  **API Layer (FastAPI):** The entry point for all client requests. It handles HTTP protocols, routing, and request/response validation using Pydantic schemas. Authentication is managed via an `X-API-Key` header.

2.  **Configuration Layer (Pydantic & YAML):** At startup, the server loads configuration from `.env` and `configs/config.yaml`. Pydantic models in `src/config.py` validate this configuration, ensuring the application starts in a known, valid state. This layer dictates which models are active, what device to use, and other core settings.

3.  **Model Management (`ModelManager`):** This central registry is responsible for the lifecycle of the ML models.
    *   It reads the model configurations.
    *   It lazy-loads models on their first request (or pre-loads all active models at startup).
    *   It ensures that each model adheres to the `BaseModel` interface defined in `src/ml/base.py`, making models interchangeable.

4.  **Inference Pipeline:**
    *   A request hits an endpoint in `/routers/analysis.py`.
    *   A FastAPI dependency (`process_video_request`) securely saves the uploaded video to a temporary file.
    *   The `ModelManager` provides the requested model instance.
    *   To avoid blocking the server's main event loop, the synchronous model inference (`model.predict_*`) is run in a separate thread using `asyncio.to_thread`.
    *   The result is formatted into a standard `APIResponse` and sent to the client.

5.  **Event Publishing (Redis):** For long analyses, models can publish progress events (e.g., "frame 50/1000 processed") to a Redis channel. This allows a separate frontend or monitoring service to provide real-time feedback to the user.

![Drishtiksha System Architecture](https://res.cloudinary.com/dcsvkcoym/image/upload/v1755760692/NTRO_Deepfake_Detection_uxuk4g.png)

---

## API Endpoints

All endpoints require an `X-API-Key` header for authentication. The primary input for analysis endpoints is `multipart/form-data` containing a `video` file.

### Status & Statistics

| Method | Path        | Description                                                              |
| ------ | ----------- | ------------------------------------------------------------------------ |
| `GET`  | `/`         | Provides a detailed health check of the service and all configured models. |
| `GET`  | `/ping`     | A simple ping endpoint to confirm the server is running. Returns `{"status": "pong"}`. |
| `GET`  | `/stats`    | Returns comprehensive server stats: device info (GPU/CPU), system info (RAM), and model details. |
| `GET`  | `/device`   | Get detailed information about the compute device.                       |
| `GET`  | `/system`   | Get system resource information (RAM, CPU, etc.).                        |
| `GET`  | `/models`   | Get detailed information about all configured and loaded models.         |
| `GET`  | `/config`   | Get a summary of the current server configuration.                       |

### Analysis

| Method | Path                         | Description                                                                                               |
| ------ | ---------------------------- | --------------------------------------------------------------------------------------------------------- |
| `POST` | `/analyze`                   | Performs a comprehensive analysis, returning a final prediction, confidence, and detailed metrics if the model supports it. |
| `POST` | `/analyze/frames`            | Performs frame-by-frame (or sequence-by-sequence) analysis for temporal inspection, returning scores for each time step. |
| `POST` | `/analyze/visualize`         | Generates and streams back a video with analysis visualizations overlaid (e.g., a real-time suspicion graph). |
| `POST` | `/analyze/comprehensive`     | A powerful single-request endpoint that can perform all analysis types at once, reusing processing to save time. |
| `POST` | `/analyze/audio`             | Performs a deepfake analysis on the audio track of a video using an audio-specific model (`SCATTERING-WAVE-V1`). |
| `GET`  | `/visualization/{filename}` | Downloads a visualization video generated by a previous `/comprehensive` or `/visualize` request. |

**Example Request (using cURL):**

```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "X-API-Key: your_secret_key" \
     -F "video=@/path/to/your/video.mp4" \
     -F "model=SIGLIP-LSTM-V3"
```

**Example Response (`/analyze`):**

```json
{
  "success": true,
  "model_used": "SIGLIP-LSTM-V3",
  "timestamp": "2025-08-21T07:10:00.123456",
  "data": {
    "prediction": "FAKE",
    "confidence": 0.975,
    "processing_time": 15.23,
    "metrics": {
      "frame_count": 50,
      "final_average_score": 0.968,
      "max_score": 0.991,
      "min_score": 0.854,
      "suspicious_frames_count": 50
    },
    "note": null
  }
}
```

---

## Analytics & Results

The server provides rich, multi-faceted results to give a deep understanding of the analysis.

*   **Overall Prediction:** A final "FAKE" or "REAL" verdict.
*   **Confidence Score:** A value from 0.0 to 1.0 indicating the model's certainty in its prediction.
*   **Per-Frame/Sequence Scores:** A list of suspicion scores for discrete segments of the video. This is crucial for identifying *which parts* of a video are suspicious.
*   **Temporal Analysis:** Metrics like rolling averages and score variance help understand the consistency of artifacts over time. A high variance might indicate sporadic glitches, whereas a consistently high score suggests a more uniform manipulation.
*   **Visualizations:** The `/visualize` endpoint generates an MP4 file with an embedded graph showing the suspicion score fluctuating over the video's duration. This provides an intuitive way to correlate visual events with model output.
*   **Audio Metrics:** For audio analysis, the API returns detailed metrics on pitch, energy, and spectral features, providing forensic-level insights into vocal patterns.