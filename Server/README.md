# Drishtiksha - Server v3.0 Technical Document

## Table of Contents

- [Getting Started](#getting-started)
- [1. Introduction & Architectural Philosophy](#1-introduction--architectural-philosophy)
  - [1.1. Core Purpose](#11-core-purpose)
  - [1.2. Key Features](#12-key-features)
  - [1.3. Architectural Pillars](#13-architectural-pillars)
- [2. System Architecture Deep Dive](#2-system-architecture-deep-dive)
  - [2.1. The Anatomy of a Request](#21-the-anatomy-of-a-request)
  - [2.2. The Model Management System](#22-the-model-management-system)
- [3. The Model Catalog](#3-the-model-catalog)
- [4. Codebase Structure & Design Patterns](#4-codebase-structure--design-patterns)
  - [4.1. Directory Deep Dive](#41-directory-deep-dive)
  - [4.2. The BaseModel Contract](#42-the-basemodel-contract)
- [5. Configuration System](#5-configuration-system)
  - [5.1. The Configuration Hierarchy](#51-the-configuration-hierarchy)
  - [5.2. Pydantic for Validation](#52-pydantic-for-validation)
- [6. API Reference](#6-api-reference)
  - [6.1. Authentication](#61-authentication)
  - [6.2. Status & Statistics Endpoints](#62-status--statistics-endpoints)
  - [6.3. Analysis Endpoint](#63-analysis-endpoint)
- [7. Containerization Strategy](#7-containerization-strategy)
  - [7.1. Docker & Docker Compose](#71-docker--docker-compose)
  - [7.2. Multi-Stage Dockerfile](#72-multi-stage-dockerfile)
- [8. Conclusion & Future Roadmap](#8-conclusion--future-roadmap)
  - [8.1. Conclusion](#81-conclusion)
  - [8.2. Future Roadmap](#82-future-roadmap)

## Getting Started

This guide provides the quickest and most reliable path for a developer to get the ML inference server running locally. The recommended and fully supported method is using **Docker and Docker Compose**.

### Prerequisites

- **Git**: To clone the repository.
- **Docker Desktop** (for Windows/Mac) or **Docker Engine + Docker Compose** (for Linux).
  - On Windows, ensure you have enabled **WSL 2 Integration** in Docker Desktop settings for your chosen Linux distribution.

### Step 1: Clone the Repository

```bash
git clone https://github.com/slantie/Drishtiksha.git
cd Drishtiksha/Server
```

### Step 2: Download Model Weights [[Google Drive]](https://drive.google.com/drive/folders/1CV2ubzgr7r7DB9uiJtpx1xV-P8pLGS8S?usp=sharing)

**This is a critical manual step.** The model weight files are large and are not tracked in the Git repository. You must acquire the necessary model files (`.pth`, `.dat`, etc.) from the provided source (e.g., a shared Google Drive link) and place them into the `Server/models/` directory.

- **To find the exact list of required files**, refer to the `model_path` and `dlib_model_path` keys for each model in the `configs/config.yaml` file.

### Step 3: Configure Your Environment

Copy the environment template to create your local configuration file.

```bash
cp .env.example .env
```

Next, open the newly created `.env` file with a text editor and configure the variables:

- `API_KEY`: **(Required)** Generate a secure, 32-character random string for authenticating requests. You can use a command like `openssl rand -hex 16` to generate one.
- `ACTIVE_MODELS`: **(Required)** A comma-separated list of model names (from `configs/config.yaml`) to load on startup. **Important: Do not use quotes and do not add spaces after the commas** (e.g., `SIGLIP-LSTM-V4,SCATTERING-WAVE-V1`).
- `DEFAULT_MODEL_NAME`: **(Required)** The model to use if none is specified in an API request. The value for this key **must be present** in your `ACTIVE_MODELS` list.
- `DEVICE`: Set to `cuda` for GPU acceleration or `cpu` for CPU-only processing.
- `REDIS_URL`: For use with Docker Compose, this should be left as the default: `redis://redis:6379`.

### Step 4: Build and Run the Service

With Docker running, use Docker Compose to build the images and start all services. This single command handles everything.

```bash
docker-compose up --build
```

- The `--build` flag tells Compose to build your application image from the `Dockerfile` the first time you run it. For subsequent starts, you can simply run `docker-compose up`.

### Step 5: Verify the Service is Running

Observe the logs in your terminal. After the model loading process completes, you should see the final confirmation messages:

```
...
22:45:15 | INFO     | src.ml.registry           | ✅ All 7 active models have been loaded successfully.
22:45:15 | INFO     | src.app.main              | ✅ Drishtiksha: Deepfake Detection startup complete. All active models loaded.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

<!-- You can now access the interactive API documentation at **http://localhost:8000/docs**. -->

<details>
  <summary><strong>Alternative: Local Development (Without Docker)</strong></summary>

This method is for developers who prefer to run the Python application directly on their host machine.

1.  **Prerequisites**:

    - **Python 3.12+**
    - **uv** (recommended) or `pip` for package management.
    - **Redis Server** installed and running on `localhost:6379`.
    - **System Build Tools**: `dlib` requires build tools.
      - On Debian/Ubuntu: `sudo apt-get install build-essential cmake`
      - On Fedora: `sudo dnf install cmake gcc-c++`

2.  **Set Up Virtual Environment & Install Dependencies**:

    ```bash
    # Create the virtual environment
    python3 -m venv .venv

    # Activate it (on Linux/macOS)
    source .venv/bin/activate

    # Install uv and sync dependencies
    pip install uv
    uv sync
    ```

3.  **Run the Server**:
    ```bash
    uv run uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
    ```

</details>

## 1. Introduction & Architectural Philosophy

### 1.1. Core Purpose

This server is the specialized Machine Learning microservice for the Drishtiksha platform. Its sole purpose is to expose a suite of powerful, enhanced deep learning models over a high-performance, secure, and robust REST API. It is architected as a stateless, computationally focused component designed for a single, critical task: receiving media files (video or audio), performing intensive AI-driven analysis, and returning structured, detailed, and forensically valuable results. It serves as the intelligent core of the platform's deepfake detection capabilities, decoupled from the main backend to allow for independent scaling and resource management.

### 1.2. Key Features

The server is engineered with a rich set of features designed for performance, extensibility, and deep forensic analysis, making it a production-grade AI inference engine.

- **Unified & Media-Agnostic API**: A single, intelligent `POST /analyze` endpoint serves as the sole entry point for all analysis requests. It dynamically routes media to the appropriate specialized models for **video** or **audio** and returns a standardized, predictable, and self-describing JSON structure (`VideoAnalysisResult` or `AudioAnalysisResult`) for each media type. This simplifies client-side integration and makes the API future-proof.

- **High-Performance Asynchronous Inference**: Built on the modern ASGI framework provided by FastAPI and Uvicorn, the server uses `asyncio.to_thread` to offload synchronous, computationally heavy model inference to a separate thread pool. This critical design pattern keeps the main event loop non-blocking, enabling high concurrency and ensuring the server remains responsive to new requests even while processing long-running analysis tasks.

- **Automated Model Discovery & Eager Loading**: The `ModelManager` component features a fully automated discovery mechanism that scans the codebase at startup to find all available model classes, completely eliminating the need for fragile, manual registration dictionaries. All models listed as active in the configuration are **eagerly loaded** into memory (CPU or GPU) before the server accepts traffic, ensuring zero "cold start" latency for all user requests.

- **Rich, Detailed Analytics**: The system is designed to provide deep analytical insights, not just a binary "REAL" or "FAKE" prediction. The API returns a wealth of data, including detailed metrics, per-frame or per-sequence suspicion scores, temporal consistency analysis, and a full suite of forensic audio metrics (pitch stability, energy, and spectral features), complete with downloadable visualization artifacts.

- **Robust Configuration & "Fail Fast" Validation**: The server's entire behavior is controlled entirely via `.env` and YAML configuration files. It leverages the Pydantic library to rigorously validate all configurations at startup. This goes beyond simple type checking to include **value validation**, such as verifying that all model weight file paths exist on disk. This "Fail Fast" approach prevents runtime errors by catching misconfigurations before the server can even start.

- **Integrated Real-time Progress Reporting**: Includes a robust, singleton Redis-based `EventPublisher`. During long analysis tasks, models can emit granular, type-safe `ProgressEvent` updates to a configurable Redis Pub/Sub channel. This is designed to feed real-time systems, such as a WebSocket server in a separate backend, providing a rich user experience.

- **Built-in System & Model Monitoring**: Exposes a suite of secure, cached endpoints (`/stats`, `/device`, `/models`) for deep, real-time observability into the server's health, detailed resource utilization (CPU/GPU memory and usage), and the precise status of every configured model in the system.

### 1.3. Architectural Pillars

The server's design is guided by a set of core software engineering principles that ensure it is robust, maintainable, and easy to extend with new AI capabilities.

- **Modularity & Extensibility (The Strategy Pattern)**: The core architecture is built around a plug-and-play model system. The `BaseModel` abstract class defines a strict contract (`load()` and `analyze()`) that all models must implement. This makes models interchangeable from the perspective of the API layer, allowing a new deepfake detection model to be added to the system with **zero changes to the core API or server logic**.

- **High Performance & Concurrency**: Built on **FastAPI**, the server leverages asynchronous request/response handling for all I/O-bound operations. More importantly, it correctly delegates synchronous, CPU/GPU-bound ML inference tasks to a separate thread pool. This prevents the primary server event loop from being blocked, allowing the service to handle a high volume of concurrent requests efficiently and without degradation in responsiveness.

- **Configuration as Code**: The server's behavior is almost entirely defined by declarative configuration files (`.env` and `config.yaml`), not imperative, hardcoded logic. Active models, compute device selection (`cpu`/`cuda`), and all model-specific hyperparameters can be changed without touching a single line of the Python source code, making the system highly adaptable to different deployment environments.

- **Strongly-Typed & Validated**: The entire application, from the lowest-level configuration values to the final API response schemas, is defined and validated using **Pydantic**. This enforces strict data integrity at every boundary, drastically reduces the potential for runtime errors, and provides excellent, self-documenting code that is easier for developers to understand and maintain.

## 2. System Architecture Deep Dive

### 2.1. The Anatomy of a Request

Understanding how the server processes an analysis request reveals its non-blocking, multi-layered, and resilient design. The entire process is orchestrated to maximize performance while ensuring stability.

![Containerization](https://raw.githubusercontent.com/slantie/Drishtiksha/main/Server/assets/Request-Lifecycle.png)

1.  **API Layer & Authentication**: A request first hits the `POST /analyze` endpoint defined in `src/app/routers/analysis.py`. The `X-API-Key` is immediately extracted from the headers and verified by the `get_api_key` security dependency. Invalid or missing keys are logged, and a `401 Unauthorized` response is sent immediately.

2.  **Dependency Injection & File Handling**: Upon successful authentication, FastAPI injects the `process_media_request` dependency. This function efficiently handles the `multipart/form-data` stream, reading the uploaded file in chunks and saving it to a secure, temporary location on disk. This approach is memory-efficient and keeps the routing logic clean and focused on orchestration.

3.  **Model Retrieval**: The endpoint requests the appropriate model (either from the form data or the configured default) from the singleton `ModelManager` instance. Since all models are eagerly loaded at startup, this is an instantaneous dictionary lookup that retrieves the fully initialized model object from the cache.

4.  **Asynchronous Execution Offload**: The core of the performance strategy lies here. The model's `analyze()` method is a standard, synchronous Python function that can take seconds or minutes to run. To prevent this from blocking FastAPI's async event loop, it is wrapped in `asyncio.to_thread()`. This delegates the long-running, synchronous inference task to a separate worker thread pool managed by `anyio`, freeing the main process to handle other requests.

5.  **Event Publishing (During Inference)**: While the `analyze()` method runs in its separate thread, it can emit progress updates using the `event_publisher`. These type-safe `ProgressEvent` objects are published to a Redis Pub/Sub channel for consumption by other microservices (e.g., a Node.js WebSocket backend).

6.  **Structured Result Generation**: The `analyze()` method completes its execution and returns a single, structured Pydantic object (`VideoAnalysisResult` or `AudioAnalysisResult`) that encapsulates every piece of data from the analysis.

7.  **Response Formatting & URL Generation**: Control returns to the main event loop. The router inspects the Pydantic result object. If a visualization was generated, it transforms the temporary file path into a fully qualified, downloadable URL. The result object is then wrapped in the standard `APIResponse` model, which adds metadata like the timestamp and model used.

8.  **Graceful Error Handling**: If at any point during this process a known error occurs (e.g., a `MediaProcessingError` from a corrupt file), a specific `try...except` block in the router catches it and raises a corresponding `HTTPException`, returning a clean and informative JSON error to the client. A global exception handler in `main.py` acts as a final safety net for any other unexpected errors.

### 2.2. The Model Management System

The `ModelManager` (`src/ml/registry.py`) is the central component of the ML architecture, designed for automation and robustness. It acts as a **Registry** and **Factory** for all models.

![Containerization](https://raw.githubusercontent.com/slantie/Drishtiksha/main/Server/assets/Model-Integration.png)

- **Automatic Discovery**: At server startup, the `ModelManager` constructor is called once. Its first action is to perform **automatic discovery**. It programmatically scans the `src/ml/models` directory and inspects each file to find all classes that inherit from our `BaseModel` contract. It builds a registry mapping the class names (e.g., `"SiglipLSTMV4"`) to the actual class objects, eliminating the need for a fragile, manually maintained registration dictionary.

- **Eager Loading on Startup**: Immediately after discovery, the FastAPI `lifespan` manager calls the `manager.load_models()` method. This method reads the list of `ACTIVE_MODELS` from the configuration. It then iterates through this list, looks up the corresponding class in the discovered registry, instantiates it with its specific Pydantic configuration object, and calls the instance's `load()` method. This is the step that loads the model weights into memory (CPU or GPU). This "warm-up" process happens entirely _before_ the server starts accepting network traffic, guaranteeing that the first user request is just as fast as any other.

- **Fail-Fast Loading**: If any model fails during the `load()` process (e.g., due to a missing weights file, a CUDA error, or a dependency issue), the `ModelManager` logs a critical error and raises a `ModelRegistryError`. This exception is caught by the `lifespan` manager, which immediately terminates the server with a non-zero exit code. This is crucial for containerized environments, as it signals to orchestrators like Kubernetes or Docker Compose that the service is unhealthy and should be restarted or investigated.

## 3. The Model Catalog

The server is designed to support a diverse suite of models, each targeting different artifacts and modalities of synthetic media. All models are defined as entries in `configs/config.yaml`, which are then parsed and validated against the Pydantic schemas in `src/config.py`. The `ModelManager` dynamically loads only the models specified in the `ACTIVE_MODELS` environment variable.

| Model Name Identifier    | Type  | Methodology & Strengths                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| :----------------------- | :---- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SIGLIP-LSTMV4**        | Video | **Enhanced Temporal Analysis**. This is the flagship video analysis model. It uses a powerful SigLIP vision transformer to extract high-quality, semantically rich features from individual frames. These feature sequences are then fed into a Bi-directional Long Short-Term Memory (Bi-LSTM) network to analyze temporal patterns and inconsistencies across the video. The V4 variant includes a deeper classifier head with dropout for improved regularization and generalization. |
| **COLOR-CUES-LSTM-V1**   | Video | **Analyzes Chromatic Inconsistencies**. This model operates on the principle that many generative models, especially older GANs, struggle with perfect color consistency in facial regions under varying lighting. It uses dlib to perform facial landmark detection, calculates chromaticity histograms for each frame's facial region, and then uses an LSTM to classify the sequence of these histograms.                                                                             |
| **EFFICIENTNET-B7-V1**   | Video | **High-Accuracy Per-Frame Detection**. This model acts as a powerful single-frame detector. It uses a highly accurate MTCNN to detect and extract all faces within each frame and then classifies each face individually using a large, pre-trained EfficientNet-B7 model. The final video-level prediction is determined by aggregating the per-frame scores using a sophisticated, confidence-based heuristic that gives more weight to high-certainty predictions.                    |
| **EYEBLINK-CNN-LSTM-V1** | Video | **Detects Unnatural Blinking Patterns**. This model targets a common physiological artifact in deepfakes: unnatural blinking. It first scans the video to identify blink events by calculating the Eye Aspect Ratio (EAR) from facial landmarks. Sequences of cropped eye regions corresponding to these blinks are then analyzed by a hybrid CNN (Xception) + LSTM architecture to determine if the pattern is consistent with natural human behavior.                                  |
| **SCATTERING-WAVE-V1**   | Audio | **Audio Deepfake & Voice Clone Detection**. This is the primary audio analysis model. It operates on the audio track of a media file, first converting it into a Mel Spectrogram image. It then uses a powerful 2D Wavelet Scattering Transform for feature extraction. This technique is robust to noise and provides stable, rich representations of the audio texture, which are then fed to a classifier to detect signs of artificial voice generation.                             |

## 4. Codebase Structure & Design Patterns

The `src/` directory is the heart of the application and is cleanly separated into two main Python packages: `app` for all web server and API-related logic, and `ml` for all machine learning components. This strict separation of concerns makes the codebase easy to navigate, test, and maintain.

```bash
Server/
├── assets
│   ├── Containerization.png
│   ├── Model-Configurations.png
│   ├── Model-Integration.png
│   └── Request-Lifecycle.png
├── configs
│   └── config.yaml
├── docker-compose.yml
├── Dockerfile
├── models
│   ├── ColorCues-LSTM-v1.pth
│   ├── EfficientNet-B7-v1
│   ├── Face-Landmarks.dat
│   ├── Frontend+Backend.txt
│   ├── Scattering-Wave-v1.pth
│   └── SigLip-LSTM-v4.pth
├── pyproject.toml
├── README.md
├── scripts
│   ├── convert_weights.py
│   └── predict.py
├── src
│   ├── app
│   │   ├── dependencies.py
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── routers
│   │   │   ├── analysis.py
│   │   │   ├── __init__.py
│   │   │   └── status.py
│   │   ├── schemas.py
│   │   └── security.py
│   ├── config.py
│   ├── __init__.py
│   ├── ml
│   │   ├── architectures
│   │   │   ├── color_cues_lstm.py
│   │   │   ├── efficientnet.py
│   │   │   ├── eyeblink_cnn_lstm.py
│   │   │   ├── __init__.py
│   │   │   ├── scattering_wave_classifier.py
│   │   │   └── siglip_lstm.py
│   │   ├── base.py
│   │   ├── event_publisher.py
│   │   ├── exceptions.py
│   │   ├── __init__.py
│   │   ├── models
│   │   │   ├── color_cues_detector.py
│   │   │   ├── efficientnet_detector.py
│   │   │   ├── eyeblink_detector.py
│   │   │   ├── __init__.py
│   │   │   ├── scattering_wave_detector.py
│   │   │   └── siglip_lstm_detector.py
│   │   ├── registry.py
│   │   ├── schemas.py
│   │   ├── system_info.py
│   │   └── utils.py
│   └── training
└── uv.lock
```

### 4.1. Directory Deep Dive

- `src/app/`: Contains all FastAPI-related code, defining the web server's behavior.

  - `routers/`: Defines the API endpoints. `analysis.py` handles the unified media processing route, while `status.py` provides the public and private monitoring endpoints.
  - `schemas.py`: Contains all Pydantic models used for request validation and, critically, for formatting the structured JSON responses. This file defines the strict data contract for the entire API.
  - `dependencies.py`: Defines reusable FastAPI dependencies, such as `process_media_request`, to handle common tasks like file uploads and model validation, keeping the routing logic clean and declarative.
  - `security.py`: Handles API key authentication for all protected endpoints.
  - `main.py`: The application's main entry point. It initializes the FastAPI app, sets up the centralized logging configuration, defines global exception handlers, and manages the startup/shutdown `lifespan` events where model loading occurs.

- `src/ml/`: Contains all machine learning logic, completely decoupled from the web layer.
  - `registry.py`: Home of the automated `ModelManager`, the central orchestrator that discovers, loads, and serves all models.
  - `base.py`: Defines the abstract `BaseModel` class. This file represents the core contract that all models must follow, making the system modular.
  - `architectures/`: Contains the raw PyTorch model definitions (the `nn.Module` classes). This is the pure deep learning code, isolated from the rest of the application's logic.
  - `models/`: Contains the high-level "handler" or "detector" classes for each model. These classes implement the `BaseModel` interface and contain the full end-to-end pipeline logic: preprocessing, calling the architecture for inference, and post-processing the results into a final Pydantic schema.
  - `utils.py`: A collection of high-performance, generic utility functions (like `extract_frames`) used by multiple models.
  - `schemas.py`: Defines Pydantic models for internal ML data structures, such as the `ProgressEvent` used by the event publisher.
  - `exceptions.py`: Defines custom, specific exceptions (`MediaProcessingError`, `InferenceError`) that models can raise to allow for granular error handling in the API layer.

### 4.2. The BaseModel Contract

The file `src/ml/base.py` defines an Abstract Base Class (`BaseModel`) that all model handlers in `src/ml/models/` must inherit from. This is a powerful implementation of the **Strategy Pattern**, which is the cornerstone of the server's extensibility.

- **The Contract**: It forces every model handler to implement a consistent, minimal interface:

  - `load()`: A method that contains all the logic for loading model weights and any other necessary artifacts (like processors or tokenizers) into memory.
  - `analyze()`: The single, unified method for performing a complete analysis on a media file. It must accept a file path and return a standardized Pydantic `AnalysisResult` object (`VideoAnalysisResult` or `AudioAnalysisResult`).

- **The Benefit**: This contract makes the models completely interchangeable from the perspective of the API layer. The `analysis.py` router can call `model.analyze()` on any model instance returned by the `ModelManager` without needing to know if it's a `SiglipLSTMV4` operating on a video or a `ScatteringWaveV1` operating on audio. This design is what makes adding new models incredibly clean and low-risk, as the core application logic never needs to be modified.

## 5. Configuration System

![Containerization](https://raw.githubusercontent.com/slantie/Drishtiksha/main/Server/assets/Model-Configurations.png)

The server employs a sophisticated, multi-layered, and self-validating configuration system designed for flexibility, clarity, and robustness. It ensures that the application is always in a valid state before it even begins to load models, preventing a wide class of runtime errors.

### 5.1. The Configuration Hierarchy

Configuration is loaded from three distinct sources, with a clear order of precedence. Settings from a higher level will always override settings from a lower level, allowing for powerful and granular control across different environments.

1.  **System Environment Variables (Highest Priority)**: These are read directly from the host environment. This is the standard method for configuring applications in containerized deployments (e.g., passing variables in `docker-compose.yml` or a Kubernetes deployment manifest).
2.  **`.env` File (Medium Priority)**: This file is used for local development and contains environment-specific settings (like `API_KEY`) and overrides for the base configuration. This file should **never** be committed to version control.
3.  **`configs/config.yaml` (Lowest Priority)**: This is the base configuration file and should be committed to version control. It defines the complete list of all available models and their static parameters (e.g., model paths, architectural details, and hyperparameters).

### 5.2. Pydantic for Validation

The `src/config.py` file is the engine of this system. It uses Pydantic's `BaseSettings` to orchestrate the loading and, crucially, the validation of all configuration at server startup.

- **Type Safety and Coercion**: Pydantic automatically ensures that all configuration values match their expected Python types (e.g., `int`, `str`, `List[str]`). If a value is of a compatible type (e.g., a port number is provided as the string `"8000"`), Pydantic will safely coerce it to the correct type (`8000` as an integer). If the type is invalid, the server will fail to start with a clear error message.

- **Value and Path Validation**: The system goes beyond type checking. It uses custom validators to perform **value validation**. The most critical example is the `ExistingPath` type, which automatically checks if the file paths provided for `model_path` and `dlib_model_path` in `config.yaml` **actually exist on the filesystem**. A missing model file is a fatal configuration error that is caught immediately at startup.

- **Cross-Field Validation**: The `Settings` class uses a `model_validator` to ensure logical consistency between different configuration values. For example, it guarantees that the `DEFAULT_MODEL_NAME` specified in the `.env` file corresponds to a model that is also listed in `ACTIVE_MODELS`, preventing a common runtime error caused by misconfiguration.

- **Discriminated Unions**: The `ModelConfig` type is an `Annotated[Union[...]]` with a `discriminator` on the `class_name` field. This is a powerful Pydantic feature that provides strict validation for the `models` section of `config.yaml`. It ensures that the configuration block for a given model contains exactly the fields required by its corresponding Python class (e.g., a `ColorCuesConfig` block _must_ contain a `dlib_model_path`, while a `SiglipLSTMv4Config` block _must_ contain a `dropout_rate`). This makes the configuration for new models self-documenting and virtually impossible to get wrong.

## 6. API Reference

The server exposes a clean, secure, and well-documented REST API. The full OpenAPI specification and interactive documentation are available at the `/docs` endpoint when the server is running.

### 6.1. Authentication

All protected endpoints require an API key to be sent in the `X-API-Key` HTTP header. Failed authentication attempts are logged for security monitoring.
`X-API-Key: your_secret_key`

### 6.2. Status & Statistics Endpoints

These endpoints are for monitoring the health and state of the service.

- `GET /` **(Public)**

  - **Description**: Provides a public health check of the service. It confirms that the service is running and lists the active models and their current load status. Ideal for use by load balancers or automated health-checking systems.
  - **Response**: `HealthStatus` object.

- `GET /ping` **(Public)**

  - **Description**: A simple, lightweight endpoint to confirm that the server is running and responsive.
  - **Response**: A JSON object: `{"status": "pong"}`.

- `GET /stats` **(Protected)**

  - **Description**: Returns a comprehensive, cached snapshot of server statistics. This includes detailed GPU/CPU information, system RAM usage, server uptime, and metadata for all available models.
  - **Response**: `ServerStats` object.

- `GET /models` **(Protected)**
  - **Description**: Returns detailed information about all models that are defined in `config.yaml` and are active in the `.env` file, including their description, compute device, and whether they are currently loaded into memory.
  - **Response**: A JSON object containing a list of `ModelInfo` objects and a summary.

### 6.3. Analysis Endpoint

This is the primary, unified endpoint for performing all deepfake detection tasks.

- `POST /analyze` **(Protected)**
  - **Description**: The single, unified endpoint for all media analysis. It accepts a media file, performs a comprehensive analysis using the specified model, and returns a rich, structured result object. The structure of the `data` field in the response is dynamic and depends on the media type analyzed.
  - **Request Body**: `multipart/form-data`
  - **Form Fields**:
    - `media` (file, **required**): The video or audio file to be analyzed.
    - `model` (string, _optional_): The name of the model to use (e.g., `SIGLIP-LSTM-V4`). If omitted, the `DEFAULT_MODEL_NAME` from the configuration is used.
    - `mediaId` (string, _optional_): A unique ID for the media, which will be included in all published Redis progress events for this task.

**Example Request (`curl`)**:

```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "X-API-Key: your_secret_key" \
     -F "media=@/path/to/your/video.mp4" \
     -F "model=SIGLIP-LSTM-V4"
```

**Example JSON Response (`VideoAnalysisResult`)**:
The `data` field will contain a `VideoAnalysisResult` object for video models.

```json
{
  "success": true,
  "model_used": "SIGLIP-LSTM-V4",
  "timestamp": "2025-08-30T10:00:00.123Z",
  "data": {
    "prediction": "FAKE",
    "confidence": 0.98,
    "processing_time": 25.4,
    "note": null,
    "media_type": "video",
    "frame_count": 500,
    "frames_analyzed": 50,
    "frame_predictions": [{ "index": 0, "score": 0.95, "prediction": "FAKE" }],
    "metrics": {
      "final_average_score": 0.97,
      "score_variance": 0.02
    },
    "visualization_path": "http://localhost:8000/analyze/visualization/some_file.mp4"
  }
}
```

**Example JSON Response (`AudioAnalysisResult`)**:
The `data` field will contain an `AudioAnalysisResult` object for audio models.

```json
{
  "success": true,
  "model_used": "SCATTERING-WAVE-V1",
  "timestamp": "2025-08-30T10:05:00.456Z",
  "data": {
    "prediction": "REAL",
    "confidence": 0.88,
    "processing_time": 2.1,
    "note": null,
    "media_type": "audio",
    "properties": {
      "duration_seconds": 2.0,
      "sample_rate": 16000,
      "channels": 1
    },
    "pitch": { "mean_pitch_hz": 150.5, "pitch_stability_score": 0.92 },
    "energy": {
      /* ... */
    },
    "spectral": {
      /* ... */
    },
    "visualization": {
      "spectrogram_url": "http://localhost:8000/analyze/visualization/spec_file.png",
      "spectrogram_data": [
        /* ... spectrogram matrix ... */
      ]
    }
  }
}
```

## 7. Containerization Strategy

The server is designed from the ground up to be deployed as a containerized microservice. The entire application stack, including its dependencies like Redis, is defined declaratively using Docker and Docker Compose, which ensures a consistent, reproducible, and portable deployment environment.

### 7.1. Docker & Docker Compose

The project includes a `docker-compose.yml` file that defines the complete, multi-service application stack. This is the recommended way to run the application for both development and production.

![Containerization](https://raw.githubusercontent.com/slantie/Drishtiksha/main/Server/assets/Containerization.png)

The Compose file orchestrates two main services:

- **`app` service**: This service builds and runs the FastAPI application. It uses the `Dockerfile` in the project root for the build instructions. It is configured to pass the `.env` file into the container to provide runtime configuration and secrets securely.
- **`redis` service**: This service runs a standard, lightweight `redis:alpine` image. It acts as the message broker for the application's `EventPublisher`.

The services are connected on a shared, isolated Docker network, which allows the `app` container to reliably connect to the `redis` container using its service name (`redis`) as the hostname. The `depends_on` directive ensures that the `app` service will not start until the `redis` service is up and healthy, guaranteeing the correct startup order.

### 7.2. Multi-Stage Dockerfile

The project includes a production-ready, multi-stage `Dockerfile` designed to build a lean, efficient, and secure final image. This approach is critical for minimizing the container's attack surface and reducing deployment times.

- **Stage 1 (The "Builder")**: This temporary build environment starts from a `python:3.12-slim-bookworm` base and installs all the system-level dependencies required for _compiling_ the Python packages (e.g., `build-essential` and `cmake` for `dlib`). It then uses `uv` to install all Python dependencies from `pyproject.toml` into a self-contained virtual environment (`.venv`).

- **Stage 2 (The "Final Image")**: This is the final, lightweight production image. It starts from the same clean `python:3.12-slim-bookworm` base. It then performs the following steps:
  1.  **Installs Runtime Dependencies**: It installs only the essential system libraries needed to _run_ the application (like `libgl1` for OpenCV and `ffmpeg` for pydub), not the large build tools.
  2.  **Creates a Non-Root User**: For enhanced security, it creates a dedicated, low-privilege `appuser` and `appgroup`.
  3.  **Copies Artifacts**: It copies the pre-built `.venv` directory from the `builder` stage and the application source code.
  4.  **Sets Permissions**: It creates a writable cache directory and ensures the `appuser` owns all the necessary files.
  5.  **Switches User**: The final `USER` instruction switches the container's execution context to the non-root `appuser`. The application will run under these restricted permissions.

This multi-stage process results in a final image that is significantly smaller and more secure than a single-stage build, as it contains no build tools, compilers, or intermediate package caches.

## 8. Conclusion & Future Roadmap

### 8.1. Conclusion

This server stands as a specialized, high-performance microservice designed for enhanced deepfake detection. Its modular and automated architecture—driven by a strict `BaseModel` contract and an auto-discovering `ModelManager`—makes it exceptionally extensible and maintainable. The robust validation, asynchronous offloading of inference, graceful error handling, and security-first containerization strategy ensure it is a reliable and future-proof core for the Drishtiksha platform's AI capabilities. Every architectural decision has been made to support stability, performance, and long-term developer productivity.

### 8.2. Future Roadmap

The current architecture provides a solid and flexible foundation that is explicitly built to evolve. The following is a roadmap of planned enhancements and future capabilities that this design readily supports.

#### **Expanding Model & Analysis Capabilities**

- **Dedicated Image-Based Deepfake Detection**

  - **Vision**: Integrate models specifically trained to identify artifacts from modern generative image models (e.g., Midjourney, DALL-E 3, Stable Diffusion). These models would look for different clues than video detectors, such as anatomical inconsistencies (e.g., extra fingers), unnatural textures, and GAN fingerprints.
  - **Implementation**: The architecture is ready for this. A new `ImageAnalysisResult` schema would be added, and a new model handler class (e.g., `StableDiffusionDetectorV1`) implementing the `BaseModel` contract would be created. After adding its configuration to `config.yaml`, the `ModelManager` would automatically make it available for use with no API changes required.

- **Generation-3 (Gen-3) Video Deepfake Detection Models**

  - **Vision**: Incorporate next-generation models designed to counter highly coherent and temporally consistent video fakes (e.g., from models like Google's Veo and OpenAI's Sora). These advanced detectors would focus less on visual glitches and more on high-level semantic inconsistencies, such as impossible physics, incorrect reflections, or unnatural character interactions.
  - **Implementation**: A new handler class implementing the `BaseModel` interface would be created. The server's ability to offload long-running inference tasks is perfectly suited for these potentially more computationally expensive models.

- **Enhanced Audio Forensics Models**
  - **Vision**: Go beyond the current audio analysis by integrating models that can detect sophisticated voice cloning. These models would analyze subtler vocal features, including prosody (rhythm, stress, intonation), emotional inconsistency between tone and content, and non-verbal artifacts like breathing patterns.
  - **Implementation**: A new model handler (e.g., `ProsodyAnalysisV1`) would be added to `src/ml/models` and registered in `config.yaml`. The unified `/analyze` endpoint would seamlessly serve it.

#### **Architectural & Performance Enhancements**

- **Model Quantization & Optimization**

  - **Vision**: Implement model quantization (e.g., INT8) and use tools like TorchScript or ONNX Runtime to further accelerate inference speed and reduce the GPU memory footprint of the models.
  - **Implementation**: This would involve adding an offline conversion step for the model weights and modifying the `load()` method in the model handlers to load the new, optimized format.

- **Batch Inference Endpoint**

  - **Vision**: Add a new endpoint that accepts multiple media files in a single request, allowing the server to process them as a batch for significantly higher GPU utilization and overall throughput.
  - **Implementation**: A new route (e.g., `/analyze/batch`) would be created. The underlying model handlers would be enhanced with an `analyze_batch` method optimized for batch processing.

- **Advanced Result Caching**
  - **Vision**: Implement a caching layer (using Redis) for analysis results of identical files. This would avoid re-computation for previously seen media, providing instantaneous responses.
  - **Implementation**: The `process_media_request` dependency would be updated to first compute a hash of the uploaded file and check Redis for an existing result before proceeding to save the file and call a model.
