# Drishtiksha - Server v3.0 Technical Document

## Table of Contents

-   [Getting Started](#getting-started)
-   [1. Introduction & Architectural Philosophy](#1-introduction--architectural-philosophy)
    -   [1.1. Core Purpose](#11-core-purpose)
    -   [1.2. Key Features](#12-key-features)
    -   [1.3. Architectural Pillars](#13-architectural-pillars)
-   [2. System Architecture Deep Dive](#2-system-architecture-deep-dive)
    -   [2.1. The Anatomy of a Request](#21-the-anatomy-of-a-request)
    -   [2.2. The Model Management System](#22-the-model-management-system)
-   [3. The Model Catalog](#3-the-model-catalog)
-   [4. Codebase Structure & Design Patterns](#4-codebase-structure--design-patterns)
    -   [4.1. Directory Deep Dive](#41-directory-deep-dive)
    -   [4.2. The BaseModel Contract](#42-the-basemodel-contract)
-   [5. Configuration System](#5-configuration-system)
    -   [5.1. The Configuration Hierarchy](#51-the-configuration-hierarchy)
    -   [5.2. Pydantic for Validation](#52-pydantic-for-validation)
-   [6. API Reference](#6-api-reference)
    -   [6.1. Authentication](#61-authentication)
    -   [6.2. Status & Statistics Endpoints](#62-status--statistics-endpoints)
    -   [6.3. Analysis Endpoints](#63-analysis-endpoints)
-   [7. Containerization Strategy](#7-containerization-strategy)
    -   [7.1. Multi-Stage Dockerfile](#71-multi-stage-dockerfile)
-   [8. Conclusion & Future Roadmap](#8-conclusion--future-roadmap)
    -   [8.1. Conclusion](#81-conclusion)
    -   [8.2. Future Roadmap](#82-future-roadmap)

## Getting Started

This guide provides the quickest path for a developer to get the ML inference server running locally.

1.  **Prerequisites**:
    *   **Python 3.12**
    *   **uv** (recommended) or `pip` for package management.
    *   **Redis Server** (for progress event publishing).
    *   **System Dependencies**: `dlib` requires build tools.
        *   On Debian/Ubuntu: `sudo apt-get install build-essential cmake`
        *   On macOS: `brew install cmake`

2.  **Clone the Repository**:
    ```bash
    mkdir Drishtiksha
    cd Drishtiksha
    git clone https://github.com/slantie/Drishtiksha.git .
    cd /Server
    ```

3.  **Set Up Virtual Environment & Install Dependencies**:
    ```bash
    # Create the virtual environment
    uv init

    # Activate it
    source .venv/bin/activate

    # Install all dependencies from uv.lock
    uv sync
    ```

4.  **Configure Environment**:
    Copy the environment template to a new `.env` file.
    ```bash
    cp .env.example .env
    ```
    Next, open the `.env` file and configure the variables:
    *   `API_KEY`: Generate a secure, 32-character random string. This is for authenticating requests from the Node.js backend.
    *   `ACTIVE_MODELS`: A comma-separated list of models from `configs/config.yaml` to load on startup (e.g., `SIGLIP-LSTM-V4,SCATTERING-WAVE-V1`).
    *   `DEVICE`: Set to `cuda` for GPU or `cpu`.

5.  **Download Model Weights**:
    This is a critical manual step. You must acquire the necessary model weight files (`.pth`, `.dat`, etc.) and place them in the `Server/models/` directory. The exact file names and paths are defined in `configs/config.yaml`.

6.  **Run the Server**:
    ```bash
    uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```
    OR
    ```bash
    python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```
    
    The server will start, pre-load all `ACTIVE_MODELS`, and be accessible at `http://localhost:8000`.

Of course. You are correct; adding a dedicated "Key Features" section will make the introduction much more impactful for anyone getting acquainted with the project. It immediately showcases the server's capabilities before diving into the architectural theory.

Here is the revised **Section 1**, with a new "Key Features" subsection integrated as requested.

---

## 1. Introduction & Architectural Philosophy

### 1.1. Core Purpose

This server is the specialized Machine Learning microservice for the Drishtiksha platform. Its sole purpose is to expose a suite of powerful, state-of-the-art deep learning models over a high-performance REST API. It is designed to be a stateless, computationally focused component that receives media files, performs intensive analysis, and returns structured, detailed results.

### 1.2. Key Features

The server is engineered with a rich set of features designed for performance, extensibility, and detailed forensic analysis:

*   **Multi-Model & Multi-Modal Engine**: Dynamically loads and serves a diverse suite of deep learning models from a central configuration file. It natively handles both **video** and **audio** analysis by routing media to the appropriate specialized models.
*   **Comprehensive Analysis API**: Offers a range of endpoints beyond simple prediction, including frame-by-frame inspection (`/analyze/frames`), generation of visual overlays (`/analyze/visualize`), and a powerful single-call comprehensive analysis (`/analyze/comprehensive`).
*   **High-Performance Asynchronous Inference**: Built on FastAPI, the server uses `asyncio.to_thread` to offload synchronous, computationally heavy model inference to a separate thread pool. This keeps the main event loop non-blocking, enabling high concurrency and responsiveness.
*   **Rich, Detailed Analytics**: Provides deep analytical insights, not just a binary "REAL" or "FAKE" prediction. The API returns detailed metrics, per-frame/sequence suspicion scores, temporal consistency analysis, and forensic audio metrics (pitch, energy, spectral features).
*   **Dynamic Configuration-Driven System**: The server's behavior—including which models are active, the compute device (`cpu` or `cuda`), and all model-specific parameters—is controlled entirely via `.env` and YAML configuration files, requiring no code changes to adapt or extend.
*   **Integrated Real-time Progress Reporting**: Includes a Redis-based event publisher (`event_publisher.py`) that emits granular progress updates during long analysis tasks. This is designed to feed the real-time WebSocket system in the Node.js backend.
*   **Built-in System & Model Monitoring**: Exposes a suite of endpoints (`/stats`, `/device`, `/models`) for deep observability into the server's health, resource utilization (CPU/GPU), and the status of every configured model.

### 1.3. Architectural Pillars

The server's design is guided by principles that ensure it is robust, maintainable, and easy to extend with new AI capabilities.

*   **Modularity & Extensibility**: The core architecture is built around a plug-and-play model system. Adding a new deepfake detection model is a predictable process that requires no changes to the core API or server logic, thanks to a standardized model interface.
*   **High Performance**: Built on **FastAPI**, the server leverages asynchronous request/response handling for I/O operations. Crucially, it offloads synchronous, CPU/GPU-bound ML inference tasks to a separate thread pool to prevent blocking the main server event loop, allowing it to handle concurrent requests efficiently.
*   **Configuration-Driven**: The server's behavior is almost entirely defined by configuration files (`.env` and `config.yaml`), not hardcoded logic. Active models, device selection, and model-specific parameters can all be changed without touching the source code.
*   **Strongly-Typed & Validated**: The entire application, from configuration to API schemas, is validated using **Pydantic**. This ensures data integrity at every level, reduces runtime errors, and provides excellent, self-documenting code.

## 2. System Architecture Deep Dive

### 2.1. The Anatomy of a Request

Understanding how the server processes an analysis request reveals its non-blocking, multi-layered design.

<!--
    FIG_REQ: FastAPI Server Request Lifecycle
    DESCRIPTION: A diagram tracing a request through the server.
    - An arrow labeled "1. HTTP POST `/analyze/comprehensive`" points to the "API Layer (FastAPI)".
    - Inside the API Layer, a box for `routers/analysis.py` receives the request.
    - An arrow from the router points to a box labeled "2. Dependency: `process_video_request`", which securely saves the uploaded file.
    - An arrow points to "3. `ModelManager.get_model('...')`", which retrieves a model instance.
    - An arrow points to a box labeled "4. `asyncio.to_thread(model.predict_detailed, ...)`". This box should have a note: "Synchronous ML inference is run in a separate thread, keeping the main event loop free."
    - Arrows trace the return path, showing the JSON response being constructed and sent back to the client.
-->

1.  **API Layer (FastAPI)**: A request first hits an endpoint defined in `src/app/routers/`. The `X-API-Key` is immediately verified by the `get_api_key` security dependency.
2.  **Dependency Injection**: FastAPI injects the `process_video_request` dependency. This function handles the boilerplate of streaming the uploaded file to a secure temporary location on disk. This approach keeps the routing logic clean and focused.
3.  **Model Retrieval**: The endpoint requests a specific model from the singleton `ModelManager` instance. If the model is not yet loaded, the manager instantiates it and loads its weights into memory (CPU or GPU RAM).
4.  **Asynchronous Execution**: The core of the performance strategy lies here. The model's `predict_*` method is a standard, synchronous Python function. To prevent it from blocking FastAPI's async event loop, it is wrapped in `asyncio.to_thread()`. This delegates the long-running, synchronous inference task to a separate thread pool managed by `anyio`.
5.  **Event Publishing (Optional)**: During inference, the model can emit progress updates using the `event_publisher`, which sends messages to a Redis Pub/Sub channel for consumption by the Node.js backend.
6.  **Response Formatting**: Once the thread completes and returns the result dictionary, the router validates this data against a Pydantic `schema`, wraps it in the standard `APIResponse` model, and sends the JSON response to the client.

### 2.2. The Model Management System

The `ModelManager` (`src/ml/registry.py`) is the central component of the ML architecture. It acts as a **Registry** and **Factory** for all models.

*   **At Startup**: The FastAPI `lifespan` manager initializes a single instance of `ModelManager`. It reads the `config.yaml` and the `ACTIVE_MODELS` from the `.env` file and pre-loads all specified models into memory. This "warm-up" ensures that the first request to any active model is fast, without lazy-loading delays.
*   **On Request**: When an endpoint requests a model, the manager checks its internal cache. If the model is already loaded, it's returned instantly. This ensures that large models are only loaded once.

## 3. The Model Catalog

The server is designed to support a diverse suite of models, each targeting different artifacts of deepfakes. All models are defined in `configs/config.yaml` and loaded dynamically by the `ModelManager`.

| Model Name | Type | Methodology & Strengths |
| :--- | :--- | :--- |
| **SIGLIP-LSTM-V4** | Video | **State-of-the-Art Temporal Analysis**. Uses a powerful SigLIP vision transformer to extract features from frames, which are fed into a Bi-LSTM to analyze temporal patterns. V4 includes a deeper classifier with dropout for improved regularization. |
| **COLOR-CUES-LSTM-V1** | Video | **Analyzes Color Inconsistencies**. Crops faces using dlib, calculates chromaticity histograms for each frame, and uses an LSTM to classify sequences of these histograms. Effective against GANs that struggle with color consistency. |
| **EFFICIENTNET-B7-V1**| Video | **High-Accuracy Frame-Level Detection**. Uses an MTCNN to find faces in each frame and then classifies each face individually using a powerful EfficientNet-B7 model. Results are aggregated using a confidence-based strategy. |
| **EYEBLINK-CNN-LSTM-V1**| Video | **Detects Unnatural Blinking Patterns**. Identifies blinks by calculating the Eye Aspect Ratio (EAR). Sequences of cropped eye regions are then analyzed by a CNN (Xception) + LSTM architecture. |
| **SCATTERING-WAVE-V1**| Audio | **Audio Deepfake & Voice Clone Detection**. Analyzes the audio track by converting it into a Mel Spectrogram and using a 2D Wavelet Scattering Transform for feature extraction, followed by a classifier. |

## 4. Codebase Structure & Design Patterns

The `src/` directory is cleanly separated into two main packages: `app` for the web server logic and `ml` for the machine learning components.

### 4.1. Directory Deep Dive

*   `src/app/`: Contains all FastAPI-related code.
    *   `routers/`: Defines the API endpoints. `analysis.py` handles all media processing routes, while `status.py` provides monitoring and health check endpoints.
    *   `schemas.py`: Contains all Pydantic models for request validation and response formatting. This enforces a strict data contract for the API.
    *   `dependencies.py`: Defines reusable dependencies, like `process_video_request`, to keep routing logic clean.
    *   `security.py`: Handles API key authentication.
    *   `main.py`: The application entry point. It initializes the FastAPI app and manages the startup/shutdown `lifespan` events.

*   `src/ml/`: Contains all machine learning logic.
    *   `registry.py`: Home of the `ModelManager`, the central orchestrator for all models.
    *   `base.py`: Defines the abstract `BaseModel` class, the contract that all models must follow.
    *   `architectures/`: Contains the raw PyTorch model definitions (the `nn.Module` classes). This is the pure deep learning code.
    *   `models/`: Contains the high-level "handler" classes for each model. These classes implement the `BaseModel` interface and contain the full pipeline logic: preprocessing, inference, and post-processing.
    *   `event_publisher.py`: A simple, robust module for publishing progress events to Redis.

### 4.2. The BaseModel Contract

The file `src/ml/base.py` defines an Abstract Base Class (`BaseModel`) that all model handlers must inherit from. This is a powerful implementation of the **Strategy Pattern**.

*   **The Contract**: It forces every model to implement a consistent interface: `load()`, `predict()`, `predict_detailed()`, `predict_frames()`, and `predict_visual()`.
*   **The Benefit**: This makes the models interchangeable from the perspective of the API layer. The `analysis.py` router can call `model.predict_detailed()` on any model instance without needing to know if it's a `SiglipLSTMV4` or a `ColorCuesLSTMV1`. This makes adding new models incredibly clean and reduces the risk of breaking existing code.

## 5. Configuration System

The server employs a sophisticated, multi-layered configuration system that provides flexibility and robustness.

### 5.1. The Configuration Hierarchy

Configuration is loaded in a specific order of precedence:

1.  **`configs/config.yaml`**: This is the base configuration file. It defines the complete list of available models and their static parameters (e.g., model paths, architectural details).
2.  **`.env` File**: This file contains environment-specific settings and secrets. **Values in this file will override values from `config.yaml`**. For example, the global `DEVICE` setting is controlled here.
3.  **Environment Variables**: System environment variables have the highest precedence and will override settings from both the `.env` file and `config.yaml`. This is standard practice for containerized deployments.

### 5.2. Pydantic for Validation

The `src/config.py` file uses Pydantic's `BaseSettings` to load and validate all configuration at startup.

*   **Type Safety**: Pydantic ensures that all configuration values match their expected types (e.g., `int`, `str`, `SecretStr` for the API key). If a value is incorrect, the server will fail to start with a clear error message, preventing runtime failures due to misconfiguration.
*   **Discriminated Unions**: The `ModelConfig` uses a Pydantic `Annotated[Union[...]]` type with a `discriminator`. This is a powerful feature that ensures that the configuration for a model in `config.yaml` perfectly matches the required fields for its specific class (e.g., a `SIGLIP-LSTM-V4` config *must* contain a `dropout_rate`).

## 6. API Reference

### 6.1. Authentication

All endpoints are protected and require an API key to be sent in the `X-API-Key` header.
`X-API-Key: your_secret_key`

### 6.2. Status & Statistics Endpoints

*   `GET /`: Provides a detailed health check of the service and the load status of all configured models.
*   `GET /stats`: Returns comprehensive server statistics, including detailed GPU/CPU information, system RAM usage, and metadata for all available models.
*   `GET /models`: Returns detailed information about all configured models, including their description, device, and whether they are currently loaded into memory.

### 6.3. Analysis Endpoints

These are the primary endpoints for performing deepfake detection. The main input is `multipart/form-data` containing a `video` file and an optional `model` field.

*   `POST /analyze`: Performs a quick but comprehensive analysis, returning a final prediction, confidence score, and detailed metrics if the model supports them.
*   `POST /analyze/frames`: Performs a frame-by-frame (or sequence-by-sequence) analysis for temporal inspection, returning scores for each time step.
*   `POST /analyze/audio`: Performs analysis specifically on the audio track of a video using an audio-focused model (e.g., `SCATTERING-WAVE-V1`).
*   `POST /analyze/comprehensive`: A powerful single-request endpoint that can perform all analysis types (detailed, frames, and visualization) at once, reusing processing to save time.
*   `POST /analyze/visualize`: Generates and streams back a video with analysis visualizations overlaid (e.g., a real-time suspicion graph).
*   `GET /visualization/{filename}`: Downloads a visualization video generated by a previous `/comprehensive` or `/visualize` request.

**Example Request (`curl`)**:
```bash
curl -X POST "http://localhost:8000/analyze/comprehensive" \
     -H "X-API-Key: your_secret_key" \
     -F "video=@/path/to/your/video.mp4" \
     -F "model=SIGLIP-LSTM-V4"
```

**Example JSON Response (`/analyze/comprehensive`)**:
```json
{
  "success": true,
  "model_used": "SIGLIP-LSTM-V4",
  "timestamp": "2025-08-22T10:00:00.123Z",
  "data": {
    "prediction": "FAKE",
    "confidence": 0.98,
    "processing_time": 25.4,
    "metrics": {
      "frame_count": 50,
      "final_average_score": 0.97,
      /* ... more metrics ... */
    },
    "frames_analysis": {
      "overall_prediction": "FAKE",
      "frame_predictions": [
        { "frame_index": 0, "score": 0.95, "prediction": "FAKE" },
        /* ... more frames ... */
      ]
    },
    "visualization_generated": true,
    "visualization_filename": "some_unique_filename.mp4",
    "processing_breakdown": {
      "basic_analysis": 15.1,
      "frames_analysis": 0.0,
      "visualization": 10.3,
      "total": 25.4
    }
  }
}
```

## 7. Containerization Strategy

### 7.1. Multi-Stage Dockerfile

The project includes a multi-stage `Dockerfile` for building lean and efficient production images.

*   **Stage 1 (Builder)**: This stage installs `uv` and all system dependencies required for building the Python packages (like `cmake` for `dlib`). It creates a virtual environment and installs all dependencies using `uv sync`.
*   **Stage 2 (Final Image)**: This stage starts from a clean `python:3.12-slim` base image. It copies the entire pre-built virtual environment from the `builder` stage, ensuring that no build tools or unnecessary files are included in the final image. This results in a smaller and more secure production container.

## 8. Conclusion & Future Roadmap

### 8.1. Conclusion

This server stands as a specialized, high-performance microservice designed for one task: state-of-the-art deepfake detection. Its modular architecture, driven by a powerful model registry and a strict `BaseModel` contract, makes it exceptionally extensible. The use of FastAPI with asynchronous offloading of ML inference ensures that the server remains responsive and capable of handling a high-throughput workload. The comprehensive, configuration-driven design allows for easy adaptation to new models and deployment environments, making it a robust and future-proof core for the Drishtiksha platform's AI capabilities.

### 8.2. Future Roadmap

The current architecture provides a solid foundation that is built to evolve. The following is a roadmap of planned enhancements and future capabilities that this design readily supports.

#### **Expanding Model & Analysis Capabilities**

*   **Dedicated Image-Based Deepfake Detection**
    *   **Vision**: Integrate models specifically trained to identify artifacts from modern generative image models (e.g., Midjourney, DALL-E 3, Stable Diffusion). These models would look for different clues than video detectors, such as anatomical inconsistencies (e.g., extra fingers), unnatural textures, and GAN fingerprints.
    *   **Implementation**: The current architecture is ready for this. A new model handler class (e.g., `StableDiffusionDetectorV1`) would be created that implements the `BaseModel` contract. After adding its configuration to `config.yaml`, the `ModelManager` would automatically make it available for use without any changes to the API layer.

*   **Generation-3 (Gen-3) Video Deepfake Detection Models**
    *   **Vision**: Incorporate next-generation models designed to counter highly coherent and temporally consistent video fakes (e.g., from models like Google's Veo 3 and ChatGPT's Sora). These advanced detectors would focus less on visual glitches and more on high-level semantic inconsistencies, such as impossible physics, incorrect reflections, or unnatural character interactions over long sequences.
    *   **Implementation**: As with other models, a new handler class implementing the `BaseModel` interface would be created. The server's ability to offload long-running inference tasks is perfectly suited for these potentially more computationally expensive models.

*   **Enhanced Audio Forensics Models**
    *   **Vision**: Go beyond the current audio analysis by integrating models that can detect sophisticated voice cloning. These models would analyze subtler vocal features, including prosody (rhythm, stress, intonation), emotional inconsistency between tone and content, and non-verbal artifacts like breathing patterns.
    *   **Implementation**: A new model handler (e.g., `ProsodyAnalysisV1`) would be added to the `ml/models` directory and registered in `config.yaml`. The existing `/analyze/audio` endpoint could be enhanced to accept a `model` parameter to allow selection between different audio analysis strategies.

#### **Architectural & Performance Enhancements**

*   **Model Quantization & Optimization**
    *   **Vision**: Implement model quantization (e.g., INT8) and use tools like TorchScript or ONNX Runtime to further accelerate inference speed and reduce the memory footprint of the models.
    *   **Implementation**: This would involve adding a conversion step for the model weights and modifying the `load()` method in the model handlers to load the optimized format.

*   **Batch Inference Endpoint**
    *   **Vision**: Add a new endpoint that accepts multiple media files in a single request, allowing the server to process them as a batch for significantly higher throughput.
    *   **Implementation**: A new route (e.g., `/analyze/batch`) would be created. The underlying model handlers would be enhanced with a `predict_batch` method optimized for batch processing.

*   **Advanced Caching**
    *   **Vision**: Implement a caching layer (using Redis) for analysis results of identical files (based on a content hash). This would avoid re-computation and provide instantaneous results for previously seen media.
    *   **Implementation**: The `process_video_request` dependency would be updated to first compute a hash of the uploaded file and check Redis for an existing result before proceeding to save the file and call the model.