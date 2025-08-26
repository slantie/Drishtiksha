# Drishtiksha - Backend v3.0 Technical Document

## Table of Contents

* [Getting Started](#getting-started)
* [1. Introduction & Architectural Philosophy](#1-introduction--architectural-philosophy)
    - [1.1. The Problem Domain](#11-the-problem-domain)
    - [1.2. Core Mandate: The System Orchestrator](#12-core-mandate-the-system-orchestrator)
    - [1.3. The Architectural Pillars](#13-the-architectural-pillars)
    * [2. System Architecture Deep Dive](#2-system-architecture-deep-dive)
    - [2.1. Component Interaction Diagram](#21-component-interaction-diagram)
    - [2.2. The Anatomy of a Request: End-to-End Data Flow](#22-the-anatomy-of-a-request-end-to-end-data-flow)
* [3. Core Technologies: Rationale and Design Choices](#3-core-technologies-rationale-and-design-choices)
* [4. Codebase Structure & Design Patterns](#4-codebase-structure--design-patterns)
    - [4.1. Directory Deep Dive](#41-directory-deep-dive)
    - [4.2. The Controller -> Service -> Repository Pattern in Action](#42-the-controller---service---repository-pattern-in-action)
    - [4.3. Error Handling Strategy](#43-error-handling-strategy)
* [5. The Asynchronous Analysis Pipeline](#5-the-asynchronous-analysis-pipeline)
    - [5.1. The Role of BullMQ: Why a Job Queue is Essential](#51-the-role-of-bullmq-why-a-job-queue-is-essential)
    - [5.2. BullMQ Flows: The Key to Multi-Model Analysis](#52-bullmq-flows-the-key-to-multi-model-analysis)
    - [5.3. The Worker Process](#53-the-worker-process-mediaworkerjs)
    - [5.4. Decoupled Real-time Feedback: The Redis Pub/Sub & Socket.IO Loop](#54-decoupled-real-time-feedback-the-redis-pubsub--socketio-loop)
* [6. Database Schema & Data Modeling Rationale](#6-database-schema--data-modeling-rationale)
    - [6.1. Guiding Principles](#61-guiding-principles)
    - [6.2. Model-by-Model Breakdown & Rationale](#62-model-by-model-breakdown--rationale)
    - [6.3. Indices and Performance](#63-indices-and-performance)
* [7. Configuration and Environment Management](#7-configuration-and-environment-management)
    - [7.1. The Role of .env](#71-the-role-of-env)
    - [7.2. The Three .env Files & Their Roles](#72-the-three-env-files--their-roles)
    - [7.3. Critical Environment Variables & Their Impact](#73-critical-environment-variables--their-impact)
* [8. Containerization & Deployment Strategy](#8-containerization--deployment-strategy)
    - [8.1. The Multi-Stage Dockerfile Explained](#81-the-multi-stage-dockerfile-explained)
    - [8.2. Docker Compose Orchestration](#82-docker-compose-orchestration)
    - [8.3. The Docker Entry Script](#83-the-docker-entry-script)
* [9. API Reference](#9-api-reference)
    - [9.1. Authentication](#91-authentication)
    - [9.2. Endpoint Deep Dive](#92-endpoint-deep-dive)
      - [Authentication Endpoints](#authentication-endpoints-apiv1auth)
      - [Media Endpoints](#media-endpoints-apiv1media)
      - [Monitoring Endpoints](#monitoring-endpoints-apiv1monitoring)
* [10. Conclusion & Future Roadmap](#10-conclusion--future-roadmap)
    - [10.1. Conclusion](#101-conclusion)
    - [10.2. Future Roadmap](#102-future-roadmap)
      - [Expanding Core Capabilities](#expanding-core-capabilities)
      - [Architectural & Performance Enhancements](#architectural--performance-enhancements)
      - [Developer Experience & Operations](#developer-experience--operations)

## Getting Started

This guide provides the quickest path for a new developer to get the project running using the recommended Docker workflow.

1.  **Prerequisites**:
    *   Node.js v18+
    *   Docker & Docker Compose

2.  **Clone the Repository**:
    ```bash
    git clone https://github.com/slantie/StarterKit.git
    cd slantie-drishtiksha/Backend
    ```

3.  **Configure Environment**:
    Copy the Docker environment template to a new `.env` file. This is pre-configured to work with the services in Docker Compose.
    ```bash
    cp .env.docker.example .env
    ```
    Next, open the `.env` file and set your unique secrets for `POSTGRES_PASSWORD`, `JWT_SECRET`, and `JWT_REFRESH_SECRET`.

4.  **Build & Run with Docker**:
    Use the provided management scripts to build the images and start all services in development mode (with hot-reloading).
    ```bash
    # For Linux/macOS
    chmod +x docker-start.sh
    ./docker-start.sh dev

    # For Windows
    .\docker-start.bat dev
    ```

5.  **Access Services**:
    *   **API Server**: `http://localhost:3000`
    *   **Prisma Studio (DB GUI)**: `http://localhost:5555`
    *   **pgAdmin (Postgres Admin)**: `http://localhost:8080`

## Available Scripts

For developers not using the Docker workflow or needing to run specific tasks, the following `npm` scripts are available in the `Backend` directory:

-   `npm run dev`: Starts the API server in development mode with hot-reloading via `nodemon`.
-   `npm run start`: Starts the API server in production mode.
-   `npm run worker`: Starts the background BullMQ worker process to handle analysis jobs.
-   `npm run dev:full`: A convenience script that runs the server (`dev`), worker, and Prisma Studio concurrently.
-   `npm test`: Executes the full test suite using Jest.
-   `npm run test:integration`: Runs the dedicated integration test runner with pre-flight checks.
-   `npm run db:generate`: Generates the Prisma Client based on your schema.
-   `npm run db:push`: Pushes the current Prisma schema state to the database (for development).
-   `npm run db:studio`: Opens the Prisma Studio web GUI to view and manage your data.

## 1. Introduction & Architectural Philosophy

### 1.1. The Problem Domain

The core challenge of deepfake analysis is not merely the AI model itself, but the construction of a system that can reliably and efficiently handle computationally expensive, long-running tasks. A naive, synchronous API would be unworkable; a user uploading a 60-second video would face a request timeout long before the analysis completes, leading to a frustrating and broken user experience.

The Drishtiksha backend is engineered to solve this specific problem. It is designed from the ground up to manage and orchestrate complex, asynchronous workflows, ensuring that the system remains responsive, scalable, and resilient, regardless of the number of concurrent users or the duration of the analysis tasks.

### 1.2. Core Mandate: The System Orchestrator

This backend is not an AI service; it is the **central orchestrator** of the entire Drishtiksha ecosystem. Its primary mandate is to serve as the robust, decoupled layer between any user-facing client (web, mobile, etc.) and the specialized Python-based Machine Learning service.

Its responsibilities include:
*   Providing a secure and stable REST API for user management, authentication, and media uploads.
*   Abstracting the complexity of the AI pipeline from the client.
*   Intelligently queuing and managing resource-intensive analysis jobs.
*   Persisting all user data, media metadata, and detailed analysis results with transactional integrity.
*   Delivering real-time, granular feedback to the user on the status of their long-running tasks.

### 1.3. The Architectural Pillars

The design of this backend is founded on four key principles. Understanding these pillars is essential to understanding the "why" behind the structure of the code and the choice of technologies.

<!--
    FIG_REQ: High-Level Architecture Diagram
    DESCRIPTION: A comprehensive diagram illustrating the primary components:
    - Client (Browser)
    - Backend API Server (Node.js)
    - Backend Worker (Node.js)
    - Redis (showing two uses: BullMQ Job Queue and Pub/Sub Channel)
    - PostgreSQL Database
    - ML Service (Python)
    Arrows should clearly label the communication protocols (e.g., REST API, WebSocket, Redis Commands, HTTP to ML).
-->

#### **Pillar 1: Asynchronous-First Processing**

*   **Rationale**: The single most important design choice is to never block the API with long-running tasks. The analysis of a single media file can take anywhere from a few seconds to many minutes. An asynchronous architecture is the only viable solution to provide a fluid user experience.
*   **Implementation**: We employ a dedicated job queue system, **BullMQ**, backed by **Redis**. When a user uploads a file, the API's only job is to validate the request, store the file, create an initial database record, and place a job (or a flow of jobs) onto the queue. It then immediately returns a `202 Accepted` response. The heavy lifting is handled entirely by separate, non-blocking worker processes.
*   **Benefit**: The user receives instant feedback that their request has been accepted. The API remains highly available and responsive to other requests, even under heavy processing load.

#### **Pillar 2: Scalability & Decoupling**

*   **Rationale**: The load characteristics of the API server and the analysis workers are fundamentally different. The API server is I/O-bound (handling many HTTP requests), while the workers are CPU/GPU-bound (handling intensive computations via the ML service). Tightly coupling them would create a monolithic application that is difficult to scale efficiently.
*   **Implementation**: The **API Server** and the **Worker** are designed as completely separate, independently deployable processes (or Docker containers). They do not communicate directly; their only link is the Redis server, which acts as a message broker.
*   **Benefit**: This decoupling allows for independent scaling. If analysis jobs are piling up, we can scale up the number of worker containers without touching the API servers. This improves resource utilization, enhances fault tolerance (a worker crash will not bring down the API), and simplifies maintenance.

#### **Pillar 3: Extensibility**

*   **Rationale**: The field of AI is constantly evolving. New models, new analysis types, and even new media types (e.g., audio, documents) will emerge. The architecture must be flexible enough to incorporate these future requirements without a complete rewrite.
*   **Implementation**:
    1.  **Dynamic Model Discovery**: The backend does not hardcode the ML models it can use. Instead, it queries the ML service's `/stats` endpoint to get a live list of available and loaded models before creating analysis jobs.
    2.  **Polymorphic Data Model**: The core `Media` table in the database is generic, using a `mediaType` enum (`VIDEO`, `AUDIO`, `IMAGE`) to differentiate uploads. Type-specific metadata and analysis results are stored in separate, related tables.
    3.  **Abstracted Storage**: The `StorageManager` allows the underlying file storage system to be swapped (e.g., from local disk to Cloudinary) via a single environment variable, without any changes to the business logic.
*   **Benefit**: This makes the system future-proof. Adding a new video analysis model requires **zero changes** to the backend codebase. Adding support for a new media type is a predictable process of adding a new service path and database tables, rather than refactoring the entire system.

#### **Pillar 4: Traceability & Observability**

*   **Rationale**: In a distributed, asynchronous system, understanding the state and history of a task is critical for debugging, monitoring, and providing user trust. A "fire-and-forget" approach is not acceptable.
*   **Implementation**:
    1.  **Granular Status Tracking**: The `MediaStatus` enum (`QUEUED`, `PROCESSING`, `ANALYZED`, `PARTIALLY_ANALYZED`, `FAILED`) provides a clear, persistent state for every uploaded file.
    2.  **Rich Data Persistence**: Every single analysis attempt is recorded in the `DeepfakeAnalysis` table. Failures are captured in the `AnalysisError` table, and detailed metadata about the model (`ModelInfo`) and system environment (`SystemInfo`) are stored with each result.
    3.  **Real-time Eventing**: The entire process is transparent to the client via a real-time feedback loop using **Socket.IO** and a **Redis Pub/Sub** channel, broadcasting granular progress events.
*   **Benefit**: This provides a complete audit trail for every job. It enables powerful debugging, allows for the creation of rich monitoring dashboards, and, most importantly, allows the frontend to display a precise and trustworthy progress status to the end-user.

## 2. System Architecture Deep Dive

This section provides a granular view of the system's components, their interactions, and the lifecycle of a request as it travels through the architecture.

### 2.1. Component Interaction Diagram

The backend is a distributed system composed of several key services that work in concert. Understanding their roles and communication paths is fundamental to understanding the system's behavior.

<!--
    FIG_REQ: High-Level Architecture Diagram
    DESCRIPTION: A comprehensive diagram illustrating the primary components:
    - Client (Browser)
    - Backend API Server (Node.js)
    - Backend Worker (Node.js)
    - Redis (showing two uses: BullMQ Job Queue and Pub/Sub Channel)
    - PostgreSQL Database
    - ML Service (Python)
    Arrows should clearly label the communication protocols (e.g., REST API, WebSocket, Redis Commands, HTTP to ML).
-->

*   **Client (Frontend)**: The user-facing application. It communicates with the backend via a REST API for state-changing actions (like uploads) and listens for real-time updates on a WebSocket connection.
*   **Backend API Server (Node.js/Express)**: The public-facing entry point. Its sole responsibilities are to handle incoming HTTP requests, perform authentication and validation, and enqueue jobs. **It performs no computationally intensive work.** It is a lightweight, I/O-optimized service.
*   **Backend Worker (Node.js/BullMQ)**: An independent, background process. Its only job is to listen for and process jobs from the Redis queue. This is the component that communicates with the heavyweight ML Service. Multiple instances of the worker can run concurrently.
*   **PostgreSQL Database**: The single source of truth for the application. It stores all persistent data, including user information, media metadata, and every detailed result from the analysis jobs. It is accessed exclusively through the Prisma ORM.
*   **Redis Server**: A multi-purpose, in-memory data store that serves two critical, distinct functions:
    1.  **Job Queue Broker**: As the backend for BullMQ, it reliably stores and manages the queue of analysis jobs waiting to be processed by the workers.
    2.  **Pub/Sub Messaging Bus**: It acts as a high-speed, fire-and-forget messaging channel (`media-progress-events`). Workers publish progress updates to this channel, and the API Server subscribes to it to forward messages to clients. This decouples the workers from the API server.
*   **ML Service (Python/FastAPI)**: A completely separate microservice that exposes AI models over a REST API. It is responsible for the actual deepfake detection, receiving a media file and returning a structured JSON result. It is a state*less* service.

### 2.2. The Anatomy of a Request: End-to-End Data Flow

To illustrate how these components interact, let's trace the complete lifecycle of a successful video upload.

1.  **[Client -> API Server] - The Upload**: The user selects a video and uploads it via a `POST` request to `/api/v1/media`. The request is `multipart/form-data` and includes the JWT in the `Authorization` header.
2.  **[API Server] - Ingestion & Validation**:
    *   The request first hits the `authenticateToken` middleware, which validates the JWT.
    *   The `multer` middleware processes the file stream, saving the video to a temporary local directory (e.g., `/temp/uploads`).
    *   The request payload is validated using Zod.
3.  **[API Server -> Storage] - File Persistence**: The `mediaService` calls the `storageManager`, which moves the temporary file to its permanent location (either the local filesystem or a cloud provider like Cloudinary).
4.  **[API Server -> PostgreSQL] - Initial Record**: The `mediaRepository` creates a new record in the `Media` table with its status set to `QUEUED`.
5.  **[API Server -> ML Service & Redis] - Job Creation & Enqueuing**:
    *   The `mediaService` makes a quick, synchronous HTTP call to the ML Service's `/stats` endpoint to get a list of currently available and compatible models.
    *   It then constructs a **BullMQ Flow**: a parent "finalizer" job and multiple child jobs, one for each compatible model.
    *   This entire flow is pushed to the `media-processing` queue in Redis.
6.  **[API Server -> Client] - Immediate Feedback**: The API server immediately responds to the client's initial `POST` request with a `202 Accepted` status code and the initial `Media` object. The HTTP request is now complete.
7.  **[Worker] - Job Consumption**: A free Worker process, which has a long-running connection to Redis, pulls one of the child analysis jobs from the queue.
8.  **[Worker -> PostgreSQL & Redis Pub/Sub] - Processing Starts**:
    *   The Worker updates the `Media` record's status in PostgreSQL to `PROCESSING`.
    *   It then publishes an `ANALYSIS_STARTED` event to the `media-progress-events` Redis channel.
9.  **[Worker -> ML Service] - The Analysis**:
    *   The Worker downloads the video file if it's in cloud storage.
    *   It sends the file in an HTTP `POST` request to the appropriate endpoint on the ML Service (e.g., `/analyze/comprehensive`). This is a long-running request.
10. **[ML Service -> Worker] - The Result**: The ML Service finishes processing and returns a detailed JSON payload with the analysis results to the Worker.
11. **[Worker -> PostgreSQL & Redis Pub/Sub] - Result Persistence & Notification**:
    *   The Worker parses the JSON result and uses the `mediaRepository` to save the detailed findings into the various analysis tables (`DeepfakeAnalysis`, `FrameAnalysis`, etc.) in a single database transaction.
    *   It publishes an `ANALYSIS_COMPLETED` event to the Redis channel.
12. **[Redis Pub/Sub -> API Server -> Client] - Real-time Update**:
    *   The API Server, which is subscribed to the Redis channel, receives the `ANALYSIS_STARTED` and `ANALYSIS_COMPLETED` events.
    *   It uses Socket.IO to broadcast these events into a user-specific "room," pushing the update directly to the connected Client's browser.
13. **[Redis & Worker] - Flow Finalization**:
    *   Steps 7-12 repeat for all other child analysis jobs in the flow.
    *   Once all child jobs are complete, BullMQ automatically enqueues the parent "finalizer" job.
    *   A Worker picks up the finalizer job.
14. **[Worker -> PostgreSQL] - Final Status Update**: The finalizer worker checks the results of all child jobs. It updates the `Media` record's status in PostgreSQL to `ANALYZED` (if all succeeded), `PARTIALLY_ANALYZED` (if some succeeded), or `FAILED` (if all failed).
15. **[Worker -> Redis Pub/Sub -> API Server -> Client] - The Final Event**: The final status update triggers one last `video_update` event, which is pushed to the client, signaling that the entire workflow is complete.

## 3. Core Technologies: Rationale and Design Choices

The technologies used in this backend were not chosen arbitrarily. Each component was selected based on its specific strengths in building a modern, scalable, and maintainable asynchronous system. This section outlines the purpose of each key technology and the rationale for its inclusion over potential alternatives.

| Technology | Purpose in This Project | Alternatives Considered | Rationale & Design Choice |
| :--- | :--- | :--- | :--- |
| **Node.js / Express.js** | **API Server & Worker Runtime** | Python (Django, FastAPI), Go (Gin), Java (Spring) | **Why Node.js?** The API server's primary role is to handle a high volume of I/O-bound operations (HTTP requests, database queries, Redis commands). Node.js's non-blocking, event-driven architecture excels at this, providing high throughput with efficient resource usage. Using a single language (JavaScript/TypeScript) for both the backend and frontend also streamlines the development process. Express.js was chosen for its minimalist, unopinionated nature, providing a solid foundation without unnecessary overhead. |
| **PostgreSQL** | **Primary Relational Datastore** | MySQL, MariaDB, MongoDB (NoSQL) | **Why PostgreSQL?** The application's data is highly relational (users have media, media has analyses, etc.). PostgreSQL offers superior data integrity, robust support for transactions, and powerful features like JSONB for storing semi-structured data within a relational context. Its maturity and reliability make it a safe choice for the system's "source of truth." A NoSQL database like MongoDB was considered but rejected due to the strongly structured and relational nature of the core data model. |
| **Prisma ORM** | **Type-Safe Database Access Layer** | TypeORM, Sequelize, Knex.js | **Why Prisma?** Prisma was chosen for its exceptional developer experience and its focus on type safety. The auto-generated Prisma Client ensures that all database queries are fully typed, catching potential errors at compile time, not runtime. Its declarative schema (`schema.prisma`) is easy to read and manage, and its migration system (`prisma migrate`) is robust and straightforward. This significantly reduces boilerplate and the likelihood of common database-related bugs compared to other ORMs. |
| **Redis** | **Job Queue Broker & Pub/Sub Bus** | RabbitMQ, Apache Kafka, In-memory store | **Why Redis?** Redis provides an optimal balance of performance and features for our specific needs. It is incredibly fast for both queuing (via BullMQ) and real-time messaging (Pub/Sub). While RabbitMQ offers more complex routing options and Kafka provides a durable log, Redis's simplicity and speed were perfectly suited for our use case. An in-memory store was rejected as it would not allow for persistence or communication between separate worker/server processes. |
| **BullMQ** | **Asynchronous Job Queue System** | Bull, Bee-Queue, Kue | **Why BullMQ?** BullMQ is the modern successor to Bull, built with TypeScript and designed for high performance. Its **Flows** feature was the killer feature for our architecture, providing a native, elegant way to manage our complex multi-model analysis workflow (parent "finalizer" job with multiple child jobs). Its rich event system and robust API for managing jobs and queues made it the superior choice over older, less feature-rich libraries. |
| **Socket.IO** | **Real-time Client Communication** | Native WebSockets, Server-Sent Events (SSE) | **Why Socket.IO?** While native WebSockets are powerful, Socket.IO provides a crucial abstraction layer that handles connection negotiation, automatic reconnection, and fallback to HTTP long-polling for environments where WebSocket connections are blocked. Its concept of **Rooms** is used heavily to broadcast events only to the specific user who owns the media, simplifying the logic for user-specific notifications. This made it a more practical and resilient choice than implementing these features manually. |
| **Docker / Docker Compose** | **Containerization & Orchestration** | Bare-metal deployment, Kubernetes (K8s) | **Why Docker?** Docker solves the classic "it works on my machine" problem by creating a consistent, reproducible, and isolated environment for every service (API, Worker, Postgres, Redis). This simplifies the developer onboarding process and ensures parity between development, testing, and production. Docker Compose is used to define and orchestrate this multi-container application locally. While Kubernetes is the standard for large-scale production, Docker Compose is perfectly suited for development and smaller-scale deployments. |

## 4. Codebase Structure & Design Patterns

The backend codebase is intentionally structured to be modular, predictable, and maintainable. It adheres to well-established design patterns to enforce a clear separation of concerns, making the system easier to reason about, test, and extend.

### 4.1. Directory Deep Dive

The `src/` directory is the heart of the application. Each sub-directory has a distinct and clearly defined responsibility.

*   `src/api/`: **The HTTP Interface Layer.** This is the entry point for all incoming API requests.
    *   **`*.routes.js`**: Defines the API endpoints, the HTTP methods they respond to, and which controller functions and middleware they use.
    *   **`*.controller.js`**: Acts as the "traffic cop." Its sole job is to parse the HTTP request (`req`), pass the relevant data to the appropriate `service` layer function, and then format the result from the service into an HTTP response (`res`). Controllers contain **no business logic**.
    *   **`*.validation.js`**: Contains the Zod schemas used to validate incoming request bodies, query parameters, and URL parameters, ensuring data integrity before any logic is executed.
*   `src/services/`: **The Business Logic Layer.** This is where the core application logic resides.
    *   Services orchestrate complex operations and business workflows. For example, `media.service.js` coordinates calls to the `storageManager` to upload a file, the `mediaRepository` to create a database record, and the `queue` to dispatch analysis jobs.
    *   They are agnostic of the HTTP layer; they do not know about `req` or `res` objects. This makes them highly reusable and easy to test in isolation.
*   `src/repositories/`: **The Data Access Layer (DAL).** This layer is the sole gateway to the database.
    *   It abstracts away all direct `Prisma` queries. Instead of scattering `prisma.media.create(...)` throughout the services, we have a clean `mediaRepository.create(...)` method.
    *   **Crucial Benefit**: This pattern makes the application data-source agnostic. If we ever needed to switch from Prisma to another ORM or even a different type of database, the only part of the application that would need to be rewritten is the `repositories` layer. The `services` layer would remain unchanged.
*   `src/workers/`: **The Background Processing Layer.** This contains the code for the independent, long-running worker processes.
    *   `media.worker.js` listens for jobs on the BullMQ queue.
    *   It contains the logic for executing the job's payload, such as calling the external ML service, handling the results, and saving them to the database via the `repositories`. This is where the most resource-intensive interactions happen, safely isolated from the API server.
*   `src/config/`: **Centralized Configuration.** Contains setup and initialization logic for core infrastructure components like the database connection (Prisma), the job queue (BullMQ), and real-time sockets (Socket.IO).
*   `src/storage/`: **The Storage Abstraction Layer.**
    *   **`storage.manager.js`**: A clever dynamic importer that reads the `STORAGE_PROVIDER` environment variable and exposes the correct storage provider (`local.provider.js` or `cloudinary.provider.js`) to the rest of the application.
    *   **`*.provider.js`**: Each provider implements a consistent interface (e.g., `uploadFile`, `deleteFile`), ensuring they are interchangeable. This is a prime example of the **Strategy Pattern**.

### 4.2. The Controller -> Service -> Repository Pattern in Action

This three-tier architecture is the backbone of the application's design. It ensures a unidirectional data flow and strict separation of concerns. Let's trace a simple `GET /api/v1/media/:id` request to see it in action:

<!--
    FIG_REQ: Three-Tier Architecture Data Flow
    DESCRIPTION: A simple, clear diagram with three vertical columns labeled "Controller", "Service", and "Repository".
    - An arrow starts from the left, labeled "1. HTTP Request (`/media/:id`)", and points to the "Controller" column.
    - Inside "Controller", a box says `media.controller.js`. An arrow goes from this box to the "Service" column, labeled "2. `getMediaById(id, userId)`".
    - Inside "Service", a box says `media.service.js`. An arrow goes from this box to the "Repository" column, labeled "3. `findByIdAndUserId(id, userId)`".
    - Inside "Repository", a box says `media.repository.js (Prisma Call)`.
    - Arrows then trace the return path: Repository -> Service -> Controller, labeled "4. Return Prisma Model", "5. Return Formatted Data", "6. Return JSON Response".
-->

1.  **Request Arrives (`Controller`)**: The request hits the router in `media.routes.js`, which invokes the `getMediaById` function in `media.controller.js`. The controller extracts the `id` from `req.params` and the `userId` from the authenticated user (`req.user`).
2.  **Logic Execution (`Service`)**: The controller calls `mediaService.getMediaWithAnalyses(id, userId)`. The service contains the business logic, which in this case is to simply fetch the data. It knows nothing about HTTP.
3.  **Data Fetching (`Repository`)**: The `mediaService` calls `mediaRepository.findByIdAndUserId(id, userId)`. This is the only place in the application that constructs and executes the actual Prisma query to fetch the media and its related analyses from the database.
4.  **Return Path**: The Prisma model is returned from the repository to the service. The service returns it to the controller. The controller then wraps the data in a consistent `ApiResponse` object and sends it back to the client as a JSON response with a `200 OK` status code.

### 4.3. Error Handling Strategy

Consistent and predictable error handling is managed centrally.

*   **`ApiError` Class (`src/utils/ApiError.js`)**: A custom error class that extends the native `Error` object. It standardizes application-specific errors by including an HTTP `statusCode`, a clear `message`, and an array of `errors`. Services and controllers throw this error for predictable failures (e.g., "Not Found," "Validation Failed").
*   **`error.middleware.js`**: This is the final middleware in the Express chain. It acts as a global catch-all.
    *   If the error is an instance of `ApiError`, it sends a structured JSON response with the error's specific status code and message.
    *   If the error is an unexpected, unhandled exception, it catches it, logs it, and sends a generic `500 Internal Server Error` response, preventing stack traces from leaking to the client.

## 5. The Asynchronous Analysis Pipeline

The true power of this backend lies in its asynchronous processing pipeline. This architecture is designed to handle tasks that are both long-running and resource-intensive, ensuring the system remains responsive and reliable. The entire pipeline is orchestrated by BullMQ, a high-performance job queue system built on Redis.

### 5.1. The Role of BullMQ: Why a Job Queue is Essential

A traditional synchronous API would force a user to wait for the entire analysis process to complete before receiving a response. For a five-minute video, this would result in a guaranteed HTTP timeout and a failed request. A job queue fundamentally solves this problem by decoupling the initial request from the actual work.

*   **Decoupling**: The API Server's responsibility ends the moment it successfully adds a job to the Redis queue. It doesn't need to know how or when the job is processed.
*   **Durability**: Jobs in the Redis queue are persistent. If a worker process crashes mid-analysis, the job is not lost. Once the worker restarts (or a new one comes online), it can pick the job back up and retry.
*   **Load Management**: A queue naturally smooths out incoming traffic. If 100 users upload videos simultaneously, the jobs are queued up and processed by the available workers at a steady, manageable pace, preventing the system from being overwhelmed.

### 5.2. BullMQ Flows: The Key to Multi-Model Analysis

A simple "one video, one job" approach is insufficient for our needs. We need to run a video through *multiple* different AI models, and we only want to consider the entire process "complete" when all of those analyses are finished. This is where **BullMQ Flows** become the cornerstone of our architecture.

A flow is a powerful parent-child job structure:

<!--
    FIG_REQ: BullMQ Flow Diagram
    DESCRIPTION: A diagram illustrating the parent-child job structure.
    - A box at the top labeled "Flow Added to Queue".
    - An arrow points down to a large box labeled "Parent Job: Finalize-Analysis (ID: video123-finalizer)". This parent box should visually contain several smaller boxes.
    - Inside the parent box, there are three smaller boxes labeled:
        - "Child Job 1: Run-Single-Analysis (Model: SIGLIP-LSTM-V4)"
        - "Child Job 2: Run-Single-Analysis (Model: COLOR-CUES-LSTM-V1)"
        - "Child Job 3: Run-Single-Analysis (Model: EFFICIENTNET-B7-V1)"
    - Text below should state: "The Parent Job will only be processed after ALL Child Jobs have completed or failed."
-->

1.  **Parent Job (The Finalizer)**: When a video is uploaded, we create a single parent job (e.g., `finalize-analysis`). This job has a unique ID tied to the media ID (e.g., `mediaId-finalizer`).
2.  **Child Jobs (The Analyses)**: This parent job is created with an array of children. Each child job is a `run-single-analysis` task, containing the `mediaId` and the specific `modelName` to be used for that analysis.
3.  **Execution Logic**: BullMQ guarantees that the parent job will **only** be processed after every single one of its child jobs has either completed successfully or failed (after all retry attempts).

This pattern provides a clean, robust, and declarative way to manage our complex workflow, ensuring the final status of the media is updated accurately and only once the entire multi-model analysis is complete.

### 5.3. The Worker Process (`media.worker.js`)

The worker is an independent Node.js process with a single responsibility: to listen for and execute jobs from the `media-processing` queue.

*   **Lifecycle**: The worker maintains a long-lived connection to Redis. When it has spare capacity, it pulls the next available job from the queue.
*   **Job Execution**: Upon receiving a `run-single-analysis` job, the worker performs the following steps:
    1.  Fetches the media record from the database to get its URL.
    2.  Updates the media status to `PROCESSING`.
    3.  **Publishes** a real-time `ANALYSIS_STARTED` event to the Redis Pub/Sub channel.
    4.  Downloads the media file from storage (if not available locally).
    5.  Sends the file to the external Python ML Service via an HTTP request.
    6.  Awaits the JSON response from the ML Service.
    7.  Saves the detailed results or error information to the PostgreSQL database.
    8.  **Publishes** an `ANALYSIS_COMPLETED` event.
    9.  Cleans up any temporary files.
*   **Concurrency**: A single worker process can handle multiple jobs concurrently (the level is configurable), allowing it to process several model analyses in parallel, maximizing throughput.

### 5.4. Decoupled Real-time Feedback: The Redis Pub/Sub & Socket.IO Loop

A critical architectural challenge is providing real-time progress updates to the user without tightly coupling the API server and the workers. A direct connection (e.g., the worker making an API call back to the server) would be brittle and hard to scale. We solve this with a classic **Pub/Sub (Publisher/Subscriber)** pattern.

<!--
    FIG_REQ: Real-time Feedback Loop Diagram (This is a more detailed version of the Redis diagram from Section 3)
    DESCRIPTION: A diagram with four components: Worker, API Server, Redis, and Client.
    1. An arrow from "Worker" to "Redis" is labeled "1. `redis.publish('media-progress-events', ...)`".
    2. An arrow from "Redis" to "API Server" is labeled "2. Redis Pub/Sub delivers event". The API Server has a note: "Subscribed to 'media-progress-events'".
    3. An arrow from "API Server" to "Client" is labeled "3. `io.to(userId).emit('progress_update', ...)`". The connection is styled to look like a WebSocket connection.
-->

1.  **The Publisher (`Worker`)**: When the worker reaches a key milestone (e.g., starting an analysis, completing it), it does not know or care which users need to be notified. It simply **publishes** a JSON payload with the event details to a specific Redis channel named `media-progress-events`. This is a "fire-and-forget" operation.
2.  **The Broker (`Redis`)**: Redis instantly broadcasts this message to any and all clients currently subscribed to that channel.
3.  **The Subscriber (`API Server`)**: The API server process, upon startup, subscribes to the `media-progress-events` channel. When it receives a message, it parses the JSON payload to get the `userId` and the event data.
4.  **The Broadcaster (`Socket.IO`)**: The API server then uses Socket.IO to emit the event into a private, user-specific "room" (identified by the `userId`). Only the user who owns the media will receive the `progress_update` event on their client-side WebSocket connection.

This elegant pattern ensures that the workers and the API servers remain completely decoupled. We can add more workers or more API servers, and the real-time communication will continue to function seamlessly, mediated entirely by Redis.

Excellent. Let's proceed to the data layer. This section is crucial as it details not just the structure of the database, but the specific design decisions and their implications for the application's functionality, performance, and future extensibility.

Here is the draft for **Section 6: Database Schema & Data Modeling Rationale**.

---

## 6. Database Schema & Data Modeling Rationale

The database is the system's long-term memory and the single source of truth. The schema, defined in `prisma/schema.prisma`, is meticulously designed to be robust, normalized, and extensible. It is built on a foundation of PostgreSQL and managed via Prisma ORM.

### 6.1. Guiding Principles

*   **Normalization Over Duplication**: The schema is highly normalized to ensure data integrity and minimize redundancy. For instance, user information is stored once in the `User` table and referenced by foreign keys elsewhere. This prevents data inconsistencies and makes updates more efficient.

*   **Extensibility for the Future**: A core design principle is that the schema must easily accommodate new media types (e.g., images) and new types of analysis data without requiring disruptive changes to the core tables. This is achieved by separating generic data from type-specific data.

*   **Comprehensive Traceability**: The schema is designed to capture a rich audit trail of the entire analysis process. It stores not just the final prediction, but also detailed metrics, errors, and metadata about the environment in which the analysis was performed. This is invaluable for debugging, reproducibility, and building user trust.

### 6.2. Model-by-Model Breakdown & Rationale

Below is a detailed examination of the key models and the reasoning behind their design.

<!--
    FIG_REQ: Database Schema ERD (Entity-Relationship Diagram)
    DESCRIPTION: A clear ERD showing the key tables and their relationships.
    - `User` --< `Media` (One-to-Many)
    - `Media` --< `DeepfakeAnalysis` (One-to-Many)
    - `Media` -- `VideoMetadata` (One-to-One)
    - `Media` -- `AudioMetadata` (One-to-One)
    - `DeepfakeAnalysis` --< `FrameAnalysis` (One-to-Many)
    - `DeepfakeAnalysis` -- `AnalysisDetails` (One-to-One)
    - `DeepfakeAnalysis` -- `AudioAnalysis` (One-to-One)
    - `DeepfakeAnalysis` -- `ModelInfo` (One-to-One)
    - `DeepfakeAnalysis` -- `SystemInfo` (One-to-One)
    - `DeepfakeAnalysis` --< `AnalysisError` (One-to-Many)
    Clearly mark the foreign key relationships.
-->

#### **`Media`: The Central, Polymorphic Hub**
This is the most important table in the schema. It represents any file uploaded by a user.
*   **Design Rationale**:
    *   The `Media` table is **polymorphic**; it can represent a video, an audio file, or an image. The `mediaType` enum (`VIDEO`, `AUDIO`, `IMAGE`) is the critical discriminator field that dictates how the application logic treats this record.
    *   The `publicId` and `url` fields abstract the storage location. Whether the file is stored locally or on Cloudinary, the rest of the application interacts with it through these consistent identifiers. This was a deliberate choice to decouple the data model from the storage implementation.

#### **DeepfakeAnalysis: The Analysis Record**
This table creates the many-to-one link between a single `Media` record and the multiple analysis results it can have.
*   **Design Rationale**:
    *   **Decoupling `model` from an Enum**: The `model` field is a `String`, not a Prisma `enum`. This is a subtle but critical design decision. **Implication**: It means the backend does not need to be aware of the ML models beforehand. If the ML team deploys a new model, the backend will dynamically discover it and save its results without requiring a schema migration or a backend redeployment. This makes the system highly extensible.
    *   It stores the top-level results common to most analyses: `prediction`, `confidence`, and `processingTime`.

#### **Media Metadata Tables**
These tables store metadata that is specific to a particular media type (e.g., duration for video/audio, width/height for video/image).
*   **Design Rationale**:
    *   **Avoiding `NULL` Columns**: The alternative would be to place all possible metadata fields (duration, width, height, bitrate, channels, etc.) directly on the `Media` table. This would result in a "bloated" table where most columns are `NULL` for any given record (e.g., an audio file would have `NULL` for width and height).
    *   **Benefit**: By creating separate one-to-one relations, we ensure that the data stored is always relevant and the schema is clean and self-documenting.

#### **Analysis Result Tables**
These tables store the rich, detailed, and often voluminous data from different types of analyses.
*   **Design Rationale**:
    *   **Preventing JSON Blobs**: The alternative would be to store the entire detailed result from the ML service as a single JSON blob in the `DeepfakeAnalysis` table.
    *   **Benefit of Normalization**: By normalizing this data into separate tables, we gain immense query power and performance. For example, we can efficiently perform queries like: "For this video, find all frames where the prediction was 'FAKE' and the confidence was above 0.9," or "Calculate the average `rmsEnergy` across all audio analyses." Such operations would be extremely slow and inefficient if they required parsing large JSON blobs across thousands of rows.

#### **Data Integrity and Cascading Deletes**
*   **Rationale**: It is essential that when a user's `Media` record is deleted, all associated data is cleaned up to prevent orphaned records and maintain database hygiene.
*   **Implementation**: The schema makes strategic use of the `onDelete: Cascade` referential action on its foreign keys.
    *   **Implication**: Deleting a `Media` record will automatically trigger a cascading delete of all its child `DeepfakeAnalysis` records. In turn, deleting a `DeepfakeAnalysis` record will cascade to delete its associated `FrameAnalysis`, `AnalysisDetails`, `AnalysisError` records, and so on. This ensures complete and automatic data cleanup with a single command.

### 6.3. Indices and Performance
*   **Rationale**: As the database grows, query performance becomes critical, especially for user-facing features like a media dashboard.
*   **Implementation**:
    *   **`@@index([userId, createdAt(sort: Desc)])` on `Media`**: This composite index is a performance optimization specifically for the most common query: "fetch the most recent media for a given user." It allows the database to locate the user's data and sort it by creation date without scanning the entire table.
    *   **`@@index([mediaId, model, analysisType])` on `DeepfakeAnalysis`**: This index speeds up internal lookups to check if a specific type of analysis has already been run for a given media file and model.

## 7. Configuration and Environment Management

The backend is designed to be highly configurable, allowing it to operate seamlessly across different environments without code changes. This is achieved through a robust environment variable strategy, facilitated by the `dotenv` library.

### 7.1. The Role of .env

The `.env` file is the cornerstone of the application's configuration. It allows for the externalization of all environment-specific settings, such as database credentials, API keys, and service URLs.

*   **Rationale**: Hardcoding configuration values directly into the source code is a significant security risk and makes the application rigid. By using environment variables, we adhere to the [Twelve-Factor App methodology](https://12factor.net/config), ensuring a clean separation between code and configuration. This means the same Docker image can be promoted from a staging to a production environment simply by providing it with a different `.env` file.
*   **Loading**: The `dotenv` library is initialized at the very beginning of the application's entry point (`server.js`). It reads the key-value pairs from the `.env` file in the project's root directory and loads them into the Node.js `process.env` object, making them globally accessible throughout the application.

### 7.2. The Three .env Files & Their Roles

The repository contains three distinct `.env` files, each serving a specific purpose in the development and deployment lifecycle. The runtime `.env` file is intentionally included in `.gitignore` to prevent sensitive credentials from ever being committed to version control.

1.  **`.env.example`**:
    *   **Purpose**: This file serves as a template for developers setting up a **local, non-Dockerized** development environment.
    *   **Content**: It lists all the necessary environment variables the application requires, with placeholder or sensible default values (e.g., `PORT=3000`, `DATABASE_URL="postgres://username:password@localhost:5432/mydatabase"`).
    *   **Workflow**: A new developer clones the repository, copies this file to `.env` (`cp .env.example .env`), and fills in their local-specific credentials.
2.  **`.env.docker.example`**:
    *   **Purpose**: This file is a template specifically for the **Dockerized** environment managed by Docker Compose.
    *   **Key Difference**: The hostnames for services like the database and Redis are not `localhost`. Instead, they use the **Docker service names** defined in `docker-compose.yml`. For example, the `REDIS_URL` is `redis://redis:6379`, where `redis` is the name of the Redis service within the Docker network.
    *   **Workflow**: When preparing to run the application with Docker Compose, a developer copies this file to `.env` (`cp .env.docker.example .env`) and modifies secrets like `POSTGRES_PASSWORD` and `JWT_SECRET`.
3.  **`.env` (Untracked)**:
    *   **Purpose**: This is the actual, runtime-specific configuration file that is actively read by the application.
    *   **Security**: It is the **only** file that should contain real secrets and credentials, and it **must never** be committed to Git. Its presence in `.gitignore` ensures this.

### 7.3. Critical Environment Variables & Their Impact

Certain environment variables fundamentally alter the application's behavior and are critical to its operation.

*   `NODE_ENV`:
    *   **Description**: Sets the application's operating mode.
    *   **Impact**: When set to `production`, Express.js enables performance optimizations and disables detailed error messages. When `development`, it provides more verbose logging and debugging information. It also influences which dependencies are installed (`npm install --omit=dev`).
*   `STORAGE_PROVIDER`:
    *   **Description**: Determines the file storage backend. Accepts `local` or `cloudinary`.
    *   **Impact**: This is a powerful switch that controls the behavior of the `storage.manager.js`. If set to `local`, all uploaded files are saved to the local filesystem. If `cloudinary`, they are uploaded to the Cloudinary cloud service. This allows for easy switching between free local storage for development and scalable cloud storage for production without any code changes.
*   `SERVER_URL` & `SERVER_API_KEY`:
    *   **Description**: The connection string and authentication key for the downstream Python ML service.
    *   **Impact**: These variables are the critical link to the AI microservice. If they are not set, the `modelAnalysisService` will be disabled, and the application will be unable to perform any deepfake analysis.
*   `DATABASE_URL`:
    *   **Description**: The full connection string for the PostgreSQL database.
    *   **Impact**: This is the primary connection string used by Prisma to connect to the database. It contains the user, password, host, port, and database name.
*   `REDIS_URL`:
    *   **Description**: The connection string for the Redis server.
    *   **Impact**: This is essential for the entire asynchronous pipeline. Both BullMQ (for job queuing) and the Socket.IO Redis adapter (for Pub/Sub) rely on this URL to connect to Redis. If this is not configured, the background processing and real-time updates will fail.

## 8. Containerization & Deployment Strategy

The entire backend ecosystem is designed to be run within Docker containers, which provides a consistent and reproducible environment for development, testing, and production. This strategy, orchestrated by Docker Compose, eliminates the "it works on my machine" problem and streamlines the deployment process.

### 8.1. The Multi-Stage Dockerfile Explained

The project uses a single, powerful `Dockerfile` that employs a **multi-stage build** pattern. This is a best practice for creating lean, secure, and optimized production images.

<!--
    FIG_REQ: Multi-Stage Dockerfile Flow
    DESCRIPTION: A diagram with two large boxes side-by-side.
    - The first box is labeled "Stage 1: The 'builder' Environment". Inside, list its key contents:
        - Node.js Alpine Image
        - Build Tools (`python3`, `g++`)
        - **ALL** Node Modules (incl. `devDependencies`)
        - Generated Prisma Client (`/app/node_modules/.prisma`)
    - An arrow points from the "builder" box to the second box. The arrow is labeled "COPY --from=builder".
    - The second box is labeled "Stage 2: The Final 'production' Image". Inside, list its contents, highlighting what was copied:
        - Clean Node.js Alpine Image
        - `postgresql-client` (for migrations)
        - **ONLY Production** Node Modules (copied from builder)
        - Generated Prisma Client (copied from builder)
        - Application Source Code
        - Non-root `appuser`
    - A caption below should state: "Result: A small, secure production image without unnecessary build tools or development dependencies."
-->

#### **Stage 1: The Builder Stage**
*   **Purpose**: This stage acts as a temporary, feature-rich environment for building our application's dependencies.
*   **Rationale**:
    1.  **Dependency Caching**: By first copying only `package*.json` and running `npm install`, we leverage Docker's build cache. If the package files haven't changed, Docker will skip the lengthy `npm install` step on subsequent builds, making development iterations much faster.
    2.  **Including `devDependencies`**: This stage intentionally installs *all* dependencies, including development ones. This is because tools like `prisma` (often a `devDependency`) are required during the build process to run `npx prisma generate`, which creates the type-safe Prisma Client.
    3.  **Compiling Native Dependencies**: It installs system-level build tools like `python3`, `make`, and `g++` that are required to compile native Node.js addons used by some npm packages.

#### **Stage 2: The Production Stage**
*   **Purpose**: This is the final, lean stage that will be used to create the image for deployment.
*   **Rationale**:
    1.  **Minimal Footprint**: It starts from a fresh, clean `node:22-alpine` image, which is very small. It does not contain any of the build tools or `devDependencies` from the `builder` stage, significantly reducing the final image size and attack surface.
    2.  **Copying Pre-built Artifacts**: Instead of running `npm install` again, it uses the `COPY --from=builder` command to copy the pre-built `node_modules` and the generated Prisma client directly from the `builder` stage. This is both faster and ensures consistency.
    3.  **Security Best Practice - Non-Root User**: A dedicated, unprivileged user (`appuser`) and group (`appgroup`) are created. The `USER appuser` command switches to this user for all subsequent operations. **Implication**: If a vulnerability were ever exploited in the Node.js application, the attacker would be confined to this unprivileged user's permissions within the container, not `root`, dramatically limiting their ability to do harm.

### 8.2. Docker Compose Orchestration

Docker Compose is used to define and run the multi-container application. The configuration is split into a base file and an override file to maintain a clean separation between production-like settings and development-specific tweaks.

*   **`docker-compose.yml` (The Base)**:
    *   **Defines Core Services**: This file defines the four essential services: `backend` (the API server), `worker` (the background job processor), `redis` (for queuing and Pub/Sub), and `postgres` (the local database).
    *   **Worker Command Override**: It demonstrates a key Docker Compose feature. Both the `backend` and `worker` services are built from the **same Docker image**. However, for the `worker` service, the `command: ["dumb-init", "--", "npm", "run", "worker"]` directive overrides the default `CMD` from the Dockerfile, instructing the container to start the worker script instead of the API server.
    *   **Networking**: It creates a dedicated bridge network (`drishtiksha-net`), allowing the containers to communicate with each other using their service names as hostnames (e.g., the backend can connect to `postgres:5432`).
*   **`docker-compose.dev.yml` (The Development Override)**:
    *   **Purpose**: This file contains settings that are only desirable during active development. It is applied using the command `docker-compose -f docker-compose.yml -f docker-compose.dev.yml up`.
    *   **Volume Mounting for Hot-Reloading**: Its most critical feature is the use of `volumes`. The line `- ./src:/app/src:ro` mounts the local `src` directory on the host machine directly into the container at `/app/src`. When a developer saves a file on their local machine, the change is instantly reflected inside the running container. Paired with `nodemon`, this enables automatic server restarts on code changes, creating a seamless development feedback loop.

### 8.3. The Docker Entry Script

This shell script is the designated `ENTRYPOINT` for the Docker image. It acts as a smart wrapper that prepares the container's environment before executing the main application command (`CMD`).

*   **Rationale**: In a containerized environment, the application container might start faster than the database container. If the application tries to connect to the database before it's ready, it will crash. This script solves that race condition.
*   **Functionality**:
    1.  **Wait for Database**: It contains a loop (`until npx prisma ...`) that repeatedly tries to connect to the database. It will pause and retry every few seconds until a successful connection is established.
    2.  **Automated Migrations**: Once the database is ready, the script runs `npx prisma migrate deploy`. This is a production-safe command that applies any pending Prisma migrations. **Implication**: This automates the deployment process. When a new version of the application with a schema change is deployed, the entrypoint script automatically updates the database schema to match what the code expects, preventing runtime errors.
    3.  **Execute Main Command**: Only after the database is ready and migrated does the script use `exec "$@"` to pass control to the container's designated `CMD` (e.g., `npm run start`).

## 9. API Reference

The Drishtiksha backend exposes a versioned RESTful API. All endpoints are prefixed with `/api/v1`. The API uses standard HTTP verbs, returns JSON-formatted responses, and uses conventional HTTP status codes to indicate the success or failure of a request.

### 9.1. Authentication

The API is secured using JSON Web Tokens (JWT).

1.  **Obtaining a Token**: A client must first register and then log in via the `/auth/login` endpoint. A successful login request will return a JWT `token`.

2.  **Using the Token**: For all protected endpoints (`/media` and `/monitoring`), the client must include this token in the `Authorization` header of every subsequent request, using the `Bearer` scheme.

    **Example Header:**
    ```http
    Authorization: Bearer <your_jwt_token_here>
    ```

    Failure to provide a valid token will result in a `401 Unauthorized` response.

### 9.2. Endpoint Deep Dive

#### **Authentication Endpoints (`/api/v1/auth`)**

*   **`POST /signup`**
    *   **Description**: Registers a new user in the system.
    *   **Request Body**: `application/json`
        ```json
        {
          "email": "user@example.com",
          "password": "a-strong-password",
          "firstName": "John",
          "lastName": "Doe"
        }
        ```
    *   **Success Response (`201 Created`)**:
        ```json
        {
          "statusCode": 201,
          "data": {
            "user": {
              "id": "clx...",
              "email": "user@example.com",
              "firstName": "John",
              "lastName": "Doe",
              "role": "USER"
            },
            "token": "ey..."
          },
          "message": "User created successfully",
          "success": true
        }
        ```
    *   **Error Responses**: `400 Bad Request` (validation error), `409 Conflict` (email already exists).

*   **`POST /login`**
    *   **Description**: Authenticates a user and returns a JWT.
    *   **Request Body**: `application/json`
        ```json
        {
          "email": "user@example.com",
          "password": "a-strong-password"
        }
        ```
    *   **Success Response (`200 OK`)**: Returns the same payload structure as `/signup`.
    *   **Error Responses**: `400 Bad Request` (validation error), `401 Unauthorized` (invalid credentials).

---

#### **Media Endpoints (`/api/v1/media`)**

*   **`POST /`**
    *   **Description**: Uploads a media file (video, audio, or image) and queues it for asynchronous analysis. This is a `multipart/form-data` request.
    *   **Request Body**:
        *   `file`: The media file to be uploaded.
        *   `description` (optional): A string description for the media.
    *   **Success Response (`202 Accepted`)**:
        The server accepts the request for processing and immediately returns the initial database record for the media. The client should then listen for WebSocket events for progress updates.
        ```json
        {
          "statusCode": 202,
          "data": {
            "id": "cly...",
            "filename": "test-video.mp4",
            "url": "http://localhost:3000/media/videos/167...",
            "publicId": "videos/167...",
            "mimetype": "video/mp4",
            "size": 559410,
            "description": "E2E Video Test",
            "status": "QUEUED",
            "mediaType": "VIDEO",
            "userId": "clx..."
          },
          "message": "Media uploaded and queued for analysis.",
          "success": true
        }
        ```
    *   **Error Responses**: `400 Bad Request` (no file provided), `401 Unauthorized`, `415 Unsupported Media Type`.

*   **`GET /:id`**
    *   **Description**: Retrieves a specific media item by its ID, along with all its completed analysis results. This is the primary endpoint for polling for the final result.
    *   **Path Parameter**: `id` (string, UUID) - The ID of the media item.
    *   **Success Response (`200 OK`)**:
        A comprehensive object including the media details and a nested array of `analyses`.
        ```json
        {
          "statusCode": 200,
          "data": {
            "id": "cly...",
            "status": "ANALYZED",
            "mediaType": "VIDEO",
            // ...other media fields
            "analyses": [
              {
                "id": "clz...",
                "prediction": "FAKE",
                "confidence": 0.987,
                "model": "SIGLIP-LSTM-V4",
                "status": "COMPLETED",
                "analysisDetails": {
                    // ...detailed video metrics
                },
                "frameAnalysis": [
                    // ...array of frame-by-frame results
                ],
                "audioAnalysis": null,
                "modelInfo": {
                    // ...metadata about the model used
                }
              },
              // ...results from other models
            ]
          },
          "success": true
        }
        ```
    *   **Error Responses**: `401 Unauthorized`, `404 Not Found`.

*   **`GET /`**
    *   **Description**: Retrieves a list of all media items belonging to the authenticated user, sorted by most recent.
    *   **Success Response (`200 OK`)**: Returns an array of media objects, each with the same structure as the response from `GET /:id`.

---

#### **Monitoring Endpoints (`/api/v1/monitoring`)**

*   **`GET /server-status`**
    *   **Description**: Retrieves the live health status and statistics of the downstream Python ML service. This acts as a health check and provides observability into the AI microservice.
    *   **Success Response (`200 OK`)**:
        The response payload is a direct, camelCased representation of the JSON output from the Python server's `/stats` endpoint.
        ```json
        {
          "statusCode": 200,
          "data": {
            "serviceName": "Drishtiksha AI Service",
            "version": "2.1.0",
            "status": "running",
            "uptimeSeconds": 7200,
            "modelsInfo": [
                {
                    "name": "SIGLIP-LSTM-V4",
                    "loaded": true,
                    "device": "cuda:0",
                    "isAudio": false,
                    "isVideo": true
                },
                // ...other model info
            ],
            "deviceInfo": {
                // ...gpu details
            }
          },
          "message": "Server status retrieved successfully",
          "success": true
        }
        ```
    *   **Error Responses**: `401 Unauthorized`, `503 Service Unavailable` (if the ML service is down).

*   **`GET /queue-status`**
    *   **Description**: Provides a real-time snapshot of the BullMQ job queue's state.
    *   **Success Response (`200 OK`)**:
        ```json
        {
          "statusCode": 200,
          "data": {
            "pendingJobs": 0,
            "activeJobs": 2,
            "completedJobs": 150,
            "failedJobs": 5,
            "delayedJobs": 0
          },
          "message": "Video processing queue status retrieved successfully.",
          "success": true
        }
        ```

## 10. Conclusion & Future Roadmap

### 10.1. Conclusion

This document has provided a comprehensive technical reference for the Drishtiksha AI Backend. At its core, this system is more than just an API; it is an asynchronous, scalable, and extensible orchestration engine designed to manage complex, long-running computational workflows.

The architectural pillars of **Asynchronous-First Processing**, **Scalability through Decoupling**, **Extensibility**, and **Traceability** are not just theoretical concepts but are deeply embedded in the codebase. The strategic use of a robust job queue (BullMQ), a highly normalized database schema (Prisma/PostgreSQL), and decoupled real-time communication (Redis Pub/Sub & Socket.IO) creates a foundation that is both powerful and maintainable. This design ensures that the backend can deliver a responsive user experience while providing a resilient, future-proof platform capable of integrating with an ever-evolving landscape of AI models and media types.

### 10.2. Future Roadmap

The current architecture provides a solid foundation that is built to evolve. The following is a roadmap of planned enhancements and future capabilities that this design readily supports.

#### **Expanding Core Capabilities**

*   **Document Analysis Support (PDFs, DOCX)**
    *   **Vision**: Extend the system to accept document files. The backend would orchestrate a new type of analysis pipeline, potentially involving OCR (Optical Character Recognition), text analysis for manipulation, and image extraction for deepfake analysis on embedded visuals.
    *   **Implementation**: This would involve adding a `DOCUMENT` type to the `MediaType` enum, creating a new `DocumentMetadata` table, and developing a new worker logic path that directs these files to a specialized document analysis microservice.
*   **Advanced Image Analysis**
    *   **Vision**: Move beyond simple deepfake detection for images and integrate models for more specific tasks like object detection, facial landmark analysis, and GAN fingerprinting.
    *   **Implementation**: This would be achieved by adding new, image-specific analysis models to the ML service. The backend's dynamic model discovery would automatically make these new analysis types available without requiring code changes.
*   **Seamless Model Integration & A/B Testing**
    *   **Vision**: Enhance the system to allow for A/B testing of different AI models. Administrators could route a certain percentage of traffic to a new, experimental model to compare its performance against a stable one.
    *   **Implementation**: This would involve adding logic to the `mediaService` to select a model based on configurable routing rules before enqueuing a job.
*   **Multi-Modal Analysis**
    *   **Vision**: The ultimate evolution for deepfake detection. This involves a sophisticated analysis that correlates a video's visual track with its audio track simultaneously. For example, a model could check if a person's lip movements realistically match the spoken words in the audio.
    *   **Implementation**: This would require a new, advanced model in the ML service and a new analysis type in the backend that could potentially trigger and then synthesize the results of separate video and audio analyses.

#### **Architectural & Performance Enhancements**

*   **Transition to a More Advanced Message Broker**
    *   **Vision**: For massive-scale deployments, transition the job queue from Redis/BullMQ to an enterprise-grade message broker like **RabbitMQ** or **Apache Kafka**.
    *   **Benefit**: This would provide more complex routing capabilities, topic-based messaging, and even stronger delivery guarantees, suitable for a high-throughput enterprise environment.
*   **Auto-Scaling Workers**
    *   **Vision**: In a cloud-native environment (e.g., Kubernetes), implement auto-scaling for the worker processes. The number of active worker containers would automatically scale up or down based on the size of the job queue.
    *   **Implementation**: This would involve using a tool like KEDA (Kubernetes Event-driven Autoscaling) to monitor the Redis queue length and adjust the worker deployment's replica count accordingly.
*   **Dedicated Caching Layer**
    *   **Vision**: Implement a dedicated Redis caching layer for frequently accessed, non-critical data, such as user profiles or the results of recent `server-status` checks.
    *   **Benefit**: This would reduce load on the primary PostgreSQL database, lowering query latency for common API requests.

#### **Developer Experience & Operations**

*   **Dedicated Monitoring Dashboard**
    *   **Vision**: Build or integrate a real-time monitoring dashboard that visualizes the data provided by the `/monitoring` endpoints.
    *   **Implementation**: Use tools like **Grafana** to create dashboards that display queue size, job processing times, ML server health, and analysis success/failure rates over time.
*   **CI/CD Pipeline Automation**
    *   **Vision**: Implement a full continuous integration and deployment pipeline using tools like GitHub Actions or Jenkins.
    *   **Implementation**: The pipeline would automatically run the full test suite, build and tag Docker images, push them to a container registry, and handle automated deployments to staging and production environments.
*   **Production-Grade Secrets Management**
    *   **Vision**: For production deployments, move secrets from `.env` files to a dedicated secrets management solution.
    *   **Implementation**: Integrate with a service like **HashiCorp Vault** or **AWS Secrets Manager** to securely inject credentials into the application containers at runtime.