# System Architecture

**Drishtiksha v3.0** - Complete System Design & Component Interactions

---

## Table of Contents

- [Overview](#overview)
- [Architectural Philosophy](#architectural-philosophy)
- [High-Level System Diagram](#high-level-system-diagram)
- [Component Breakdown](#component-breakdown)
- [Data Flow Architecture](#data-flow-architecture)
- [Request Lifecycle](#request-lifecycle)
- [Real-Time Communication Architecture](#real-time-communication-architecture)
- [Database Architecture](#database-architecture)
- [Deployment Architecture](#deployment-architecture)
- [Security Architecture](#security-architecture)
- [Scalability Considerations](#scalability-considerations)

---

## Overview

Drishtiksha is built on a **modern microservices architecture** that separates concerns across three primary services, each optimized for its specific role. This design enables independent scaling, technology-specific optimization, and maintainability at scale.

The architecture follows these core principles:

- **Separation of Concerns**: Each service has a single, well-defined responsibility
- **Asynchronous Processing**: Long-running tasks never block the user-facing API
- **Decoupled Communication**: Services communicate through well-defined interfaces
- **Stateless Services**: All application state lives in PostgreSQL and Redis
- **Event-Driven Updates**: Real-time progress via publish-subscribe patterns

---

## Architectural Philosophy

### The Problem with Monolithic Design

Traditional monolithic architectures fail for AI-intensive applications because:

1. **Resource Mismatch**: API servers (I/O-bound) and AI inference (GPU-bound) have different scaling needs
2. **Blocking Operations**: Long-running AI tasks would timeout HTTP connections
3. **Technology Lock-in**: Unable to optimize each component with the best-suited technology
4. **Deployment Complexity**: A single code change requires redeploying the entire system

### The Microservices Solution

Drishtiksha solves these problems through:

1. **Service Specialization**: Each service uses the optimal technology for its workload
2. **Async-First Design**: Job queue decouples request acceptance from processing
3. **Independent Scaling**: Scale AI workers without touching API servers
4. **Technology Freedom**: Node.js for I/O, Python for ML, PostgreSQL for persistence

---

## High-Level System Diagram

> **DIAGRAM PLACEHOLDER: High-Level Architecture**
>
> **What to Include:**
> A comprehensive system overview showing all major components and their interactions.
>
> **Components to Show:**
>
> 1. **Client Layer** (Browser/Mobile)
>    - Visual: Modern device icons
>    - Connections: Bidirectional arrows to Frontend
>
> 2. **Frontend Service** (React Application)
>    - Box: Light blue color (#3B82F6)
>    - Label: "Frontend (React 19 + Vite)"
>    - Port: `:5173` (development)
>    - Connections:
>      - REST API arrows → Backend (solid line)
>      - WebSocket connection → Backend (dashed line, labeled "Real-time Updates")
>
> 3. **Backend Service** (Node.js Orchestrator)
>    - Box: Green color (#10B981)
>    - Label: "Backend API (Node.js + Express)"
>    - Port: `:3000`
>    - Internal components (smaller boxes):
>      - "API Server" (Handles HTTP)
>      - "Worker Process" (Processes jobs)
>    - Connections:
>      - Solid arrows → PostgreSQL (labeled "Prisma ORM")
>      - Solid arrows → Redis (labeled "BullMQ Jobs")
>      - Dashed arrows ← Redis (labeled "Pub/Sub Subscribe")
>      - HTTP arrows → ML Server (labeled "Analysis Requests")
>
> 4. **ML Server** (Python Inference)
>    - Box: Purple color (#8B5CF6)
>    - Label: "ML Server (Python + FastAPI)"
>    - Port: `:8000`
>    - Internal: Small GPU icon
>    - Connections:
>      - Dashed arrows → Redis (labeled "Pub/Sub Publish")
>      - Return arrows ← Backend (labeled "JSON Results")
>
> 5. **PostgreSQL Database**
>    - Box: Blue color (#2563EB)
>    - Cylinder/database icon
>    - Label: "PostgreSQL 15"
>    - Port: `:5432`
>
> 6. **Redis Cache/Queue**
>    - Box: Red color (#DC2626)
>    - Label: "Redis 7"
>    - Port: `:6379`
>    - Two internal sections:
>      - "Job Queue (BullMQ)"
>      - "Pub/Sub Channel"
>
> 7. **Storage Layer**
>    - Box: Orange color (#F59E0B)
>    - Label: "File Storage"
>    - Options shown: "Local Filesystem" OR "Cloudinary"
>    - Connections from Backend
>
> **Layout:**
>
> - Use a layered approach (top to bottom):
>   - Layer 1: Client
>   - Layer 2: Frontend
>   - Layer 3: Backend (left), ML Server (right)
>   - Layer 4: PostgreSQL (left), Redis (center), Storage (right)
>
> **Visual Style:**
>
> - Clean, modern design
> - Use icons for services (React logo, Node.js logo, Python logo, etc.)
> - Color-coded connections (green for HTTP, blue for WebSocket, red for Pub/Sub)
> - Include port numbers on each service
> - Add small descriptive labels on arrows

---

## Component Breakdown

### 1. Frontend Service (React 19)

**Technology Stack:**

- React 19 with modern hooks
- Vite for build tooling
- Tailwind CSS for styling
- React Query for server state management
- Socket.IO client for WebSocket connections

**Responsibilities:**

- **User Interface**: Render all UI components and manage user interactions
- **Authentication**: Store JWT tokens and manage session state
- **Media Upload**: Handle file selection and multipart form submission
- **Real-Time Updates**: Listen for WebSocket events and update UI reactively
- **State Management**: Manage client-side state with React Context and React Query
- **Routing**: Client-side navigation with React Router

**Key Characteristics:**

- **Stateless**: All persistent data lives in the backend
- **Reactive**: UI updates automatically when server state changes
- **Optimistic Updates**: UI responds immediately, then syncs with server
- **Error Handling**: Graceful degradation and user-friendly error messages

---

### 2. Backend Service (Node.js + Express)

**Technology Stack:**

- Node.js 18+ runtime
- Express.js web framework
- Prisma ORM for database access
- BullMQ for job queue management
- Socket.IO server for WebSocket connections
- JWT for authentication

**Responsibilities:**

#### 2a. API Server Process

- **HTTP API**: Expose RESTful endpoints for all user actions
- **Authentication**: Validate JWT tokens on protected routes
- **Request Validation**: Validate incoming data with Zod schemas
- **Media Ingestion**: Handle file uploads and store in filesystem/cloud
- **Database Operations**: CRUD operations via Prisma ORM
- **Job Enqueueing**: Create and push jobs to Redis queue
- **Real-Time Broadcast**: Forward Redis Pub/Sub events to WebSocket clients

#### 2b. Worker Process

- **Job Consumption**: Pull jobs from Redis queue using BullMQ
- **ML Service Communication**: Make HTTP requests to Python ML Server
- **Result Persistence**: Save detailed analysis results to PostgreSQL
- **Progress Publishing**: Emit events to Redis Pub/Sub channel
- **Error Handling**: Retry failed jobs with exponential backoff

**Key Characteristics:**

- **Stateless**: Each request is independent
- **Asynchronous**: Never blocks on long-running operations
- **Scalable**: Can run multiple API servers and workers independently
- **Resilient**: Automatic job retries and error recovery

**Architecture Pattern:**

```bash
Controller → Service → Repository → Prisma → PostgreSQL
```

- **Controller**: Handle HTTP request/response
- **Service**: Business logic and orchestration
- **Repository**: Database abstraction layer

---

### 3. ML Server (Python + FastAPI)

**Technology Stack:**

- Python 3.12
- FastAPI framework
- PyTorch for deep learning
- Transformers library (Hugging Face)
- OpenCV for video processing
- Librosa for audio processing
- Redis client for Pub/Sub

**Responsibilities:**

- **Model Loading**: Eagerly load all active models into memory (CPU/GPU) at startup
- **Inference Execution**: Run media files through neural networks
- **Result Formatting**: Structure predictions into Pydantic schemas
- **Progress Reporting**: Publish fine-grained progress events to Redis
- **Health Monitoring**: Expose GPU/CPU statistics and model status

**Key Characteristics:**

- **Stateless**: No data persistence (all results returned to backend)
- **GPU-Optimized**: Leverage CUDA acceleration when available
- **Async-Capable**: Use `asyncio.to_thread` for concurrent inference
- **Modular**: Auto-discovery of new models without code changes
- **Fail-Fast**: Strict validation of configuration at startup

**Model Categories:**

1. **Video Models**: Analyze temporal sequences and visual artifacts
2. **Audio Models**: Process spectrograms and acoustic features
3. **Image Models**: Single-frame or image-specific detectors
4. **Multi-Modal Models**: Combined audio-visual analysis

---

### 4. PostgreSQL Database

**Version:** PostgreSQL 15

**Schema Design Principles:**

- **Normalization**: Minimize data redundancy
- **Referential Integrity**: Foreign keys with cascade deletes
- **Indexing**: Optimize common query patterns
- **JSONB**: Store flexible, semi-structured data

**Core Tables:**

- `users`: User accounts and authentication
- `media`: Uploaded media files metadata
- `analysis_runs`: User-initiated analysis executions
- `deepfake_analyses`: Individual model results
- `server_health`: ML server status snapshots

**Performance Features:**

- Composite indexes on `(userId, createdAt)` for fast user queries
- JSONB indexes for querying nested analysis results
- Automatic cleanup via cascading deletes

---

### 5. Redis Cache & Queue

**Version:** Redis 7

**Dual Purpose:**

#### 5a. Job Queue (BullMQ)

- **Queue Name**: `media-processing`
- **Job Types**:
  - `run-single-analysis`: Execute one model on one media file
  - `finalize-analysis`: Aggregate results after all models complete
- **Features**:
  - Persistent jobs (survive server restarts)
  - Priority queues
  - Delayed jobs
  - Automatic retries with exponential backoff

#### 5b. Pub/Sub Messaging

- **Channel Name**: `media-progress-events`
- **Event Types**:
  - `ANALYSIS_STARTED`: Worker begins processing
  - `ANALYSIS_PROGRESS`: Incremental updates (e.g., "Processing frame 50/500")
  - `ANALYSIS_COMPLETED`: Model finishes successfully
  - `ANALYSIS_FAILED`: Error occurred
- **Pattern**: Fire-and-forget, high-throughput messaging

---

### 6. Storage Layer

**Dual Strategy:**

#### Option 1: Local Filesystem

- **Path**: `Backend/database/media/{videos|audios|images}/`
- **Use Case**: Development, intranet deployments
- **Pros**: Free, no external dependencies
- **Cons**: Not scalable, no CDN

#### Option 2: Cloudinary

- **Integration**: Via `cloudinary` npm package
- **Use Case**: Production, cloud deployments
- **Pros**: Scalable, CDN-backed, automatic optimization
- **Cons**: Costs, external dependency

**Switching**: Controlled entirely by `STORAGE_PROVIDER` environment variable

---

## Data Flow Architecture

### 1. Media Upload Flow

> **DIAGRAM PLACEHOLDER: Upload Flow Sequence Diagram**
>
> **What to Include:**
> A sequence diagram showing the complete upload and analysis workflow.
>
> **Participants (Left to Right):**
>
> 1. User (stick figure icon)
> 2. Frontend (React logo)
> 3. Backend API (Node.js logo)
> 4. Storage (folder/cloud icon)
> 5. PostgreSQL (database icon)
> 6. Redis Queue (Redis logo)
> 7. Worker Process (cog icon)
> 8. ML Server (Python logo)
>
> **Sequence of Interactions:**
>
> 1. User → Frontend: "Selects video file"
> 2. Frontend → Backend API: "POST /api/v1/media (multipart/form-data)"
> 3. Backend API → Backend API: "Validate JWT token"
> 4. Backend API → Backend API: "Validate file type & size"
> 5. Backend API → Storage: "Upload file (local or Cloudinary)"
> 6. Storage → Backend API: "Return file URL & public ID"
> 7. Backend API → PostgreSQL: "INSERT INTO media (status: QUEUED)"
> 8. PostgreSQL → Backend API: "Return media record with ID"
> 9. Backend API → ML Server: "GET /stats (fetch available models)"
> 10. ML Server → Backend API: "Return list of loaded models"
> 11. Backend API → Backend API: "Create BullMQ Flow (parent + children jobs)"
> 12. Backend API → Redis Queue: "Enqueue analysis jobs"
> 13. Backend API → Frontend: "202 Accepted (media record)"
> 14. Frontend → User: "Show 'Upload successful, analyzing...'"
>
> **Parallel Processing (Loop):**
>
> 15. Redis Queue → Worker Process: "Pull next job from queue"
> 16. Worker Process → PostgreSQL: "UPDATE media SET status = 'PROCESSING'"
> 17. Worker Process → Redis Pub/Sub: "PUBLISH 'ANALYSIS_STARTED' event"
> 18. Worker Process → ML Server: "POST /analyze (send media file)"
> 19. ML Server → ML Server: "Load model, run inference"
> 20. ML Server → Redis Pub/Sub: "PUBLISH 'ANALYSIS_PROGRESS' events"
> 21. ML Server → Worker Process: "Return JSON result"
> 22. Worker Process → PostgreSQL: "INSERT INTO deepfake_analyses (result)"
> 23. Worker Process → Redis Pub/Sub: "PUBLISH 'ANALYSIS_COMPLETED' event"
> 24. Worker Process → Redis Queue: "Mark job as completed"
>
> **Finalization:**
>
> 25. Redis Queue → Worker Process: "All child jobs done, pull parent job"
> 26. Worker Process → PostgreSQL: "UPDATE media SET status = 'ANALYZED'"
> 27. Worker Process → Redis Pub/Sub: "PUBLISH final status event"
>
> **Real-Time Updates (Parallel Stream):**
>
> 28. Redis Pub/Sub → Backend API: "Subscribe receives events"
> 29. Backend API → Frontend: "Socket.IO emit to user's room"
> 30. Frontend → User: "Update progress bar & UI in real-time"
>
> **Visual Style:**
>
> - Use activation boxes (vertical rectangles) to show when each service is active
> - Dotted return arrows for responses
> - Color-code arrows: Green (HTTP), Blue (Database), Red (Pub/Sub), Orange (WebSocket)
> - Add timing notes (e.g., "~30-60 seconds" on ML Server processing)
> - Highlight the asynchronous boundary (step 13 where user gets immediate response)

---

## Request Lifecycle

### Step-by-Step Breakdown

#### Phase 1: Upload & Validation (< 1 second)

1. **User Action**: User selects a video file in the browser
2. **Client Validation**: Frontend checks file size and type
3. **HTTP Request**: POST multipart/form-data to `/api/v1/media`
4. **Authentication**: Backend validates JWT from `Authorization` header
5. **Middleware Processing**:
   - `multer` middleware streams file to temporary directory
   - Zod validation ensures request schema is correct
6. **File Storage**: `StorageManager` moves file to permanent location
7. **Database Record**: Prisma creates `Media` record with `status: QUEUED`
8. **Immediate Response**: API returns `202 Accepted` with media object

**Key Point**: The HTTP request completes here. User is not blocked.

---

#### Phase 2: Job Enqueueing (< 100ms)

1. **Model Discovery**: Backend queries ML Server's `/stats` endpoint
2. **Compatibility Filtering**: Identifies models compatible with media type
3. **Flow Construction**:
   - Creates a parent "finalizer" job
   - Creates multiple child "analysis" jobs (one per model)
4. **Queue Push**: BullMQ adds the entire flow to Redis queue
5. **WebSocket Notification**: Backend emits `upload_complete` event to user

---

#### Phase 3: Asynchronous Processing (30s - 5min)

**Worker picks up first child job:**

1. **Job Consumption**: Worker pulls job from Redis (FIFO order)
2. **Status Update**: Updates `Media.status = 'PROCESSING'` in database
3. **Progress Event**: Publishes `ANALYSIS_STARTED` to Redis channel
4. **File Download**: If using cloud storage, downloads file locally
5. **HTTP Call**: Worker POSTs file to ML Server's `/analyze` endpoint
6. **Inference**: ML Server runs the model (this is the long step)
7. **Progress Updates**: ML Server publishes granular progress events
8. **Result Reception**: Worker receives JSON response from ML Server
9. **Database Save**: Worker creates `DeepfakeAnalysis` record with full result
10. **Completion Event**: Publishes `ANALYSIS_COMPLETED` to Redis channel
11. **Job Completion**: BullMQ marks job as completed

**This repeats for all child jobs in parallel** (limited by worker concurrency)

---

#### Phase 4: Finalization (< 100ms)

**After all child jobs complete:**

1. **Parent Job Trigger**: BullMQ automatically enqueues the parent job
2. **Worker Picks Up**: Worker pulls the finalizer job
3. **Result Aggregation**: Worker queries all child job results from database
4. **Status Determination**:
   - All succeeded → `ANALYZED`
   - Some failed → `PARTIALLY_ANALYZED`
   - All failed → `FAILED`
5. **Final Update**: Updates `Media.status` with final state
6. **Final Event**: Publishes `video_update` event with complete results
7. **WebSocket Push**: Frontend receives event and updates UI

---

## Real-Time Communication Architecture

### The Pub/Sub Pattern

> **DIAGRAM PLACEHOLDER: Real-Time Event Flow**
>
> **What to Include:**
> A focused diagram showing how real-time updates flow from worker to user.
>
> **Components (Circular Flow):**
>
> 1. **Worker Process** (bottom left)
>    - Icon: Cog/gear
>    - Action: "Completes analysis step"
>
> 2. **Redis Pub/Sub** (center)
>    - Icon: Redis logo with "Pub/Sub" label
>    - Channel: `media-progress-events`
>    - Shows message structure:
>
>      ```json
>      {
>        "type": "ANALYSIS_STARTED",
>        "mediaId": "abc123",
>        "userId": "user456",
>        "modelName": "SIGLIP-LSTM-V4",
>        "timestamp": "2025-10-26T10:00:00Z"
>      }
>      ```
>
> 3. **Backend API Server** (top center)
>    - Icon: Node.js logo
>    - Label: "Subscribed to channel"
>    - Action: "Receives all events"
>
> 4. **Socket.IO Rooms** (top right)
>    - Icon: Network/broadcast symbol
>    - Shows multiple rooms:
>      - Room: `user-123` (User A)
>      - Room: `user-456` (User B)
>      - Room: `user-789` (User C)
>    - Highlight: Only matching userId gets the event
>
> 5. **Frontend Client** (right)
>    - Icon: Browser window
>    - Label: "Connected WebSocket"
>    - Shows event handler:
>
>      ```javascript
>      socket.on('progress_update', (data) => {
>        // Update UI
>      })
>      ```
>
> 6. **UI Update** (bottom right)
>    - Icon: Progress bar or spinner
>    - Shows visual update in real-time
>
> **Flow Arrows:**
>
> - Worker → Redis: "PUBLISH event" (solid red arrow)
> - Redis → Backend: "Broadcast to subscribers" (dashed red arrow)
> - Backend → Backend: "Filter by userId" (internal arrow)
> - Backend → Socket.IO Room: "Emit to specific room" (solid blue arrow)
> - Socket.IO → Frontend: "WebSocket message" (dashed blue arrow)
> - Frontend → UI: "React state update" (internal arrow)
>
> **Visual Style:**
>
> - Use a circular/cyclical layout to show the continuous flow
> - Highlight the "filtering" step where only the relevant user gets the event
> - Show example event payloads in small code boxes
> - Use different colors for Pub (red) vs WebSocket (blue) communication

### Why This Pattern?

**Problem:** Workers and API servers are separate processes, possibly on different machines.

**Solution:** Redis Pub/Sub acts as a message broker:

1. **Decoupling**: Workers don't need to know which API server has the user's WebSocket
2. **Scalability**: Works with multiple API servers (all subscribe to the channel)
3. **Reliability**: Redis handles message delivery
4. **Performance**: Fire-and-forget publishing is extremely fast

**Event Types:**

| Event Type | Published By | Triggered When | Payload |
|------------|--------------|----------------|---------|
| `ANALYSIS_STARTED` | Worker | Job begins processing | `mediaId`, `modelName`, `userId` |
| `ANALYSIS_PROGRESS` | ML Server | Incremental progress | `mediaId`, `progress`, `total` |
| `ANALYSIS_COMPLETED` | Worker | Model finishes | `mediaId`, `result`, `confidence` |
| `ANALYSIS_FAILED` | Worker | Error occurs | `mediaId`, `error`, `message` |
| `video_update` | Worker | Final status change | `mediaId`, `status`, `analysisCount` |

---

## Database Architecture

### Schema Design Philosophy

The database is designed with three core principles:

1. **Normalization**: Minimize redundancy, ensure referential integrity
2. **Flexibility**: Use JSONB for semi-structured data that evolves
3. **Performance**: Strategic indexing for common query patterns

---

### Entity Relationship Diagram

> **DIAGRAM PLACEHOLDER: Database ERD**
>
> **What to Include:**
> A comprehensive Entity-Relationship Diagram showing all tables and relationships.
>
> **Tables to Show (with key fields):**
>
> 1. **users**
>    - PK: `id` (UUID)
>    - Fields: `email`, `firstName`, `lastName`, `password`, `role`, `isActive`
>    - Timestamps: `createdAt`, `updatedAt`
>
> 2. **media**
>    - PK: `id` (UUID)
>    - FK: `userId` → users.id
>    - Fields: `filename`, `url`, `publicId`, `mimetype`, `size`, `status`, `mediaType`, `hasAudio`
>    - JSON: `metadata` (FFprobe data)
>    - FK: `latestAnalysisRunId` → analysis_runs.id (nullable)
>    - Timestamps: `createdAt`, `updatedAt`
>
> 3. **analysis_runs**
>    - PK: `id` (UUID)
>    - FK: `mediaId` → media.id
>    - Fields: `runNumber`, `status`
>    - Unique: `(mediaId, runNumber)`
>    - Timestamp: `createdAt`
>
> 4. **deepfake_analyses**
>    - PK: `id` (UUID)
>    - FK: `analysisRunId` → analysis_runs.id
>    - Fields: `modelName`, `prediction`, `confidence`, `status`, `errorMessage`
>    - Promoted: `processingTime`, `mediaType`
>    - JSON: `resultPayload` (full ML Server response)
>    - Timestamps: `createdAt`, `updatedAt`
>
> 5. **server_health**
>    - PK: `id` (UUID)
>    - Fields: `status`, `responseTimeMs`
>    - JSON: `statsPayload` (full /stats response)
>    - Timestamp: `checkedAt`
>
> **Relationships:**
>
> - users ──< media (one-to-many, cascade delete)
> - media ──< analysis_runs (one-to-many, cascade delete)
> - analysis_runs ──< deepfake_analyses (one-to-many, cascade delete)
> - media ── analysis_runs (one-to-one, nullable, "latest run")
>
> **Indexes to Highlight:**
>
> - `users.email` (unique)
> - `media(userId, createdAt DESC)` (composite, for user dashboard)
> - `deepfake_analyses(analysisRunId)` (foreign key lookup)
> - `deepfake_analyses(modelName)` (filtering by model)
> - `deepfake_analyses(confidence)` (sorting by confidence)
>
> **Visual Style:**
>
> - Use standard ERD notation (crow's foot for relationships)
> - Color-code tables by domain:
>   - Blue: Authentication (users)
>   - Green: Media (media, analysis_runs)
>   - Purple: Results (deepfake_analyses)
>   - Orange: Monitoring (server_health)
> - Show cardinality (1:N, 1:1)
> - Highlight JSONB fields with special icon
> - Show indexes with small index icon next to field names

---

### Key Design Decisions

#### 1. Analysis Versioning

**Problem**: Users may want to re-run analysis as models improve.

**Solution**: The `AnalysisRun` table:

- Each upload gets a `media` record (permanent)
- Each user-initiated analysis creates an `analysis_run` record
- `runNumber` increments (Run 1, Run 2, etc.)
- Each run can have multiple `deepfake_analyses` (one per model)

**Benefit**: Historical tracking, A/B testing of models, audit trail

---

#### 2. JSONB for Flexibility

**Problem**: ML Server response schemas evolve with new models.

**Solution**: Store the entire response in `deepfake_analyses.resultPayload`:

```json
{
  "prediction": "FAKE",
  "confidence": 0.98,
  "processing_time": 25.4,
  "media_type": "video",
  "frame_count": 500,
  "frames_analyzed": 50,
  "frame_predictions": [...],
  "metrics": {...},
  "visualization_path": "..."
}
```

**Benefit**: Backend doesn't need schema changes when new models are added

**Trade-off**: Less queryable than normalized tables, but we promote critical fields (`processingTime`, `confidence`) for filtering

---

#### 3. Cascading Deletes

**Problem**: Deleting a user should clean up all their data.

**Solution**: Foreign keys with `onDelete: Cascade`:

```bash
DELETE users WHERE id = '123'
  ↓
  Automatically deletes all media for that user
    ↓
    Automatically deletes all analysis_runs for that media
      ↓
      Automatically deletes all deepfake_analyses for those runs
```

**Benefit**: No orphaned records, simplified cleanup logic

---

## Deployment Architecture

### Docker Compose Setup

> **DIAGRAM PLACEHOLDER: Docker Deployment**
>
> **What to Include:**
> A visual representation of the Docker Compose orchestration.
>
> **Docker Network** (large container representing isolated network):
>
> - Label: `drishtiksha-net` (bridge network)
>
> **Containers (Inside Network):**
>
> 1. **frontend** (Container box)
>    - Image: Custom build from `Frontend/Dockerfile`
>    - Ports: `5173:5173`
>    - Volumes: `./Frontend/src:/app/src` (for hot reload)
>    - Env: `VITE_BACKEND_URL=http://backend:3000`
>
> 2. **backend** (Container box)
>    - Image: Custom build from `Backend/Dockerfile`
>    - Ports: `3000:3000`
>    - Volumes: `./Backend/src:/app/src` (for hot reload)
>    - Depends on: `postgres`, `redis`
>    - Command: `npm run dev`
>
> 3. **worker** (Container box)
>    - Image: Same as backend (shared image)
>    - Ports: None (internal only)
>    - Volumes: Same as backend
>    - Depends on: `postgres`, `redis`
>    - Command: `npm run worker` (overrides default)
>
> 4. **server** (Container box)
>    - Image: Custom build from `Server/Dockerfile`
>    - Ports: `8000:8000`
>    - Volumes: `./Server/models:/app/models` (model weights)
>    - Depends on: `redis`
>    - GPU: Optional GPU passthrough
>
> 5. **postgres** (Container box)
>    - Image: `postgres:15-alpine`
>    - Ports: `5432:5432`
>    - Volumes: `postgres-data:/var/lib/postgresql/data` (persistent)
>    - Env: `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`
>
> 6. **redis** (Container box)
>    - Image: `redis:7-alpine`
>    - Ports: `6379:6379`
>    - Volumes: `redis-data:/data` (persistent)
>
> **Volume Labels** (Outside network, connected with dotted lines):
>
> - `postgres-data` (disk icon)
> - `redis-data` (disk icon)
>
> **Connection Arrows:**
>
> - frontend → backend (labeled "HTTP")
> - backend → postgres (labeled "Prisma")
> - backend → redis (labeled "BullMQ")
> - worker → redis (labeled "Queue Consumer")
> - worker → server (labeled "HTTP Analysis Request")
> - server → redis (labeled "Pub/Sub Publish")
>
> **Visual Style:**
>
> - Use container icons (Docker whale logo)
> - Color-code by service type (same colors as high-level diagram)
> - Show shared volumes with dotted lines
> - Indicate port mappings (host:container)
> - Highlight that backend and worker use the same image but different commands

### Multi-Stage Dockerfile Pattern

Both Backend and Server use multi-stage builds for optimization:

#### Stage 1: Builder

- Full development dependencies
- Build tools (gcc, python, etc.)
- Compile native modules
- Generate Prisma client

#### Stage 2: Production

- Minimal base image (alpine)
- Copy only production dependencies
- Copy built artifacts from Stage 1
- Non-root user for security
- Smaller image size (~50% reduction)

---

## Security Architecture

### Authentication Flow

> **DIAGRAM PLACEHOLDER: Authentication Flow**
>
> **What to Include:**
> A sequence diagram showing JWT-based authentication.
>
> **Sequence:**
>
> 1. User → Frontend: "Enter email + password"
> 2. Frontend → Backend: "POST /api/v1/auth/login"
> 3. Backend → PostgreSQL: "SELECT * FROM users WHERE email = ?"
> 4. PostgreSQL → Backend: "Return user record"
> 5. Backend → Backend: "bcrypt.compare(password, hashedPassword)"
> 6. Backend → Backend: "jwt.sign({ userId, role }, JWT_SECRET)"
> 7. Backend → Frontend: "Return { token, user }"
> 8. Frontend → Frontend: "localStorage.setItem('token', ...)"
> 9. Frontend → User: "Redirect to dashboard"
>
> **Protected Request:**
>
> 10. User → Frontend: "Upload media"
> 11. Frontend → Backend: "POST /api/v1/media (Authorization: Bearer `token`)"
> 12. Backend → Backend: "jwt.verify(token, JWT_SECRET)"
> 13. Backend → Backend: "Attach userId to req.user"
> 14. Backend → PostgreSQL: "INSERT INTO media (userId = req.user.id)"
> 15. Backend → Frontend: "202 Accepted"
>
> **Visual Style:**
>
> - Highlight the JWT signing step (step 6)
> - Show token storage in browser (localStorage icon)
> - Use padlock icons for secure operations
> - Color-code: Green for successful auth, Red for failed

### Security Layers

1. **Transport Security**:
   - HTTPS in production (TLS 1.3)
   - Secure WebSocket (WSS)

2. **Authentication**:
   - JWT tokens with expiration
   - bcrypt password hashing (10 rounds)
   - API key authentication for ML Server

3. **Authorization**:
   - Role-based access control (USER/ADMIN)
   - User data isolation (can only access own media)

4. **Input Validation**:
   - Zod schemas for all API inputs
   - File type and size restrictions
   - SQL injection prevention (Prisma parameterized queries)

5. **Container Security**:
   - Non-root users in all containers
   - Read-only filesystem where possible
   - Minimal base images (Alpine Linux)

6. **Secrets Management**:
   - Environment variables for all secrets
   - Never commit `.env` files
   - Docker secrets in production

---

## Scalability Considerations

### Horizontal Scaling Strategies

#### 1. Frontend Scaling

**Method**: Multiple instances behind a load balancer (Nginx, HAProxy)

**Considerations**:

- Stateless (no server-side sessions)
- CDN for static assets
- Build-time environment variable injection

---

#### 2. Backend API Scaling

**Method**: Multiple instances with shared Redis/PostgreSQL

**Considerations**:

- Stateless API servers
- Socket.IO with Redis adapter (for WebSocket scaling)
- Session affinity not required
- Health checks for load balancer

**Example - 3 API Servers**:

```bash
Load Balancer
    ├── API Server 1 (Port 3000)
    ├── API Server 2 (Port 3001)
    └── API Server 3 (Port 3002)
         ↓
    Shared PostgreSQL
    Shared Redis
```

---

#### 3. Worker Scaling

**Method**: Add more worker processes/containers

**Considerations**:

- Workers automatically coordinate via Redis queue
- No communication between workers
- Linear scaling with number of workers
- Monitor queue depth to auto-scale

**Example - 5 Workers**:

```bash
Redis Queue (media-processing)
    ├── Worker 1 (concurrency: 5)
    ├── Worker 2 (concurrency: 5)
    ├── Worker 3 (concurrency: 5)
    ├── Worker 4 (concurrency: 5)
    └── Worker 5 (concurrency: 5)
         ↓
    ML Server (shared)
```

**Total Concurrent Jobs**: 25 (5 workers × 5 concurrency)

---

#### 4. ML Server Scaling

**Method**: Multiple instances with round-robin or model-specific routing

**Considerations**:

- GPU assignment (each instance can use different GPU)
- Model loading time (keep instances warm)
- Load balancer with health checks
- Consider dedicated instances per model type

**Example - GPU-Optimized Setup**:

```bash
Backend Workers
    ↓ (Round-robin)
    ├── ML Server 1 (GPU 0: Video models)
    ├── ML Server 2 (GPU 1: Video models)
    ├── ML Server 3 (CPU: Audio models)
    └── ML Server 4 (GPU 2: Image models)
```

---

### Database Scaling

**Current**: Single PostgreSQL instance

**Future Options**:

1. **Read Replicas**: Separate read/write traffic
2. **Connection Pooling**: PgBouncer for connection management
3. **Partitioning**: Partition `deepfake_analyses` by date
4. **Sharding**: Shard by `userId` for multi-tenant scaling

---

### Bottleneck Analysis

> **DIAGRAM PLACEHOLDER: Performance Bottlenecks**
>
> **What to Include:**
> A chart showing where bottlenecks occur under load.
>
> **Graph Type**: Horizontal bar chart
>
> **Bottlenecks (from most to least critical):**
>
> 1. **ML Inference** (Longest bar - red)
>    - Label: "30-60 seconds per video"
>    - Solution: "Scale ML Servers, GPU acceleration"
>
> 2. **Database Writes** (Medium bar - orange)
>    - Label: "High volume of analysis results"
>    - Solution: "Connection pooling, batch inserts"
>
> 3. **Redis Queue** (Short bar - yellow)
>    - Label: "Queue depth under heavy load"
>    - Solution: "Add more workers, prioritize jobs"
>
> 4. **File Storage** (Short bar - yellow)
>    - Label: "I/O for large video files"
>    - Solution: "Cloud storage with CDN"
>
> 5. **API Throughput** (Shortest bar - green)
>    - Label: "Rarely a bottleneck (async design)"
>    - Solution: "Add API servers if needed"
>
> **Visual Style:**
>
> - Color gradient from red (critical) to green (not critical)
> - Include "Current Capacity" and "Under 10x Load" annotations
> - Show mitigation strategies next to each bar

---

## Monitoring & Observability

### Health Check Endpoints

| Endpoint | Purpose | Response Time |
|----------|---------|---------------|
| `GET /api/health` | Backend API health | < 50ms |
| `GET /api/v1/monitoring/server-status` | ML Server health | < 200ms |
| `GET /api/v1/monitoring/queue-status` | Queue depth & stats | < 100ms |
| `GET /stats` (ML Server) | Detailed server stats | < 500ms |

### Metrics to Monitor

1. **API Server**:
   - Request rate (req/sec)
   - Response time (p50, p95, p99)
   - Error rate (%)
   - Active WebSocket connections

2. **Workers**:
   - Jobs processed/minute
   - Job failure rate
   - Average job duration
   - Queue depth

3. **ML Server**:
   - GPU utilization (%)
   - GPU memory usage
   - Inference time per model
   - Model load status

4. **Database**:
   - Active connections
   - Query latency
   - Slow query log
   - Database size

5. **Redis**:
   - Memory usage
   - Pub/Sub messages/sec
   - Queue depth by queue

---

## Technology Decision Matrix

| Decision | Options Considered | Chosen | Rationale |
|----------|-------------------|---------|-----------|
| **API Framework** | Express, Fastify, NestJS | Express | Mature, unopinionated, extensive ecosystem |
| **ORM** | Prisma, TypeORM, Sequelize | Prisma | Type-safety, excellent DX, migration tooling |
| **Job Queue** | BullMQ, Bull, Bee-Queue | BullMQ | Modern, TypeScript, **Flows feature** critical |
| **Database** | PostgreSQL, MySQL, MongoDB | PostgreSQL | JSONB, reliability, complex queries |
| **ML Framework** | FastAPI, Flask, Django | FastAPI | Async support, Pydantic, performance |
| **Real-time** | Socket.IO, Native WS, SSE | Socket.IO | Fallback support, rooms, reliability |
| **Storage** | Local, S3, Cloudinary | Cloudinary | CDN, transformations, easy integration |

---

## Conclusion

The Drishtiksha architecture is designed for:

- **Performance**: Asynchronous processing, GPU acceleration, efficient queuing
- **Scalability**: Independent scaling of each service tier
- **Reliability**: Job persistence, automatic retries, health monitoring
- **Maintainability**: Clear separation of concerns, type safety, modular design
- **Extensibility**: Auto-discovery patterns, pluggable storage, JSONB flexibility

This architecture supports the current workload while providing clear paths for future enhancements, including Kubernetes orchestration, advanced analytics, and multi-region deployment.

---

**Next Steps:**

- [Backend Architecture Deep Dive](/docs/backend/Architecture)
- [ML Server Architecture Deep Dive](/docs/server/Architecture)
- [Deployment Guide](/docs/Deployment/Intranet-Guide)
