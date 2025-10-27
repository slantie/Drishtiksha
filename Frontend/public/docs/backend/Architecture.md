# Backend Architecture

**Drishtiksha Backend** is a Node.js microservice built with **Express.js** that serves as the orchestration layer between the React frontend and the Python ML inference engine. It handles user authentication, media management, job scheduling, real-time progress updates, and PDF report generation.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Technology Stack](#technology-stack)
3. [Architecture Layers](#architecture-layers)
4. [Microservices Communication](#microservices-communication)
5. [Request Flow](#request-flow)
6. [Design Patterns](#design-patterns)
7. [Service Configuration](#service-configuration)
8. [Background Workers](#background-workers)
9. [Error Handling & Resilience](#error-handling--resilience)
10. [Deployment Architecture](#deployment-architecture)

---

## System Overview

### High-Level Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                        Drishtiksha Platform                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────────┐      ┌───────────────┐      ┌─────────────┐ │
│  │   Frontend    │◄────►│    Backend    │◄────►│   Server    │ │
│  │  (React 19)   │      │  (Express.js) │      │  (FastAPI)  │ │
│  │               │      │               │      │             │ │
│  │ • UI/UX       │      │ • Auth        │      │ • ML Models │ │
│  │ • WebSocket   │      │ • API Gateway │      │ • Inference │ │
│  │ • State Mgmt  │      │ • Job Queue   │      │ • Progress  │ │
│  └───────────────┘      └───────────────┘      └─────────────┘ │
│         │                       │                      │         │
│         │                       │                      │         │
│         └───────────────────────┴──────────────────────┘         │
│                                 │                                │
│                    ┌────────────┴────────────┐                  │
│                    │                         │                  │
│              ┌─────▼──────┐           ┌─────▼──────┐           │
│              │ PostgreSQL │           │   Redis    │           │
│              │  (Prisma)  │           │ (Pub/Sub)  │           │
│              └────────────┘           └────────────┘           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Core Responsibilities

- **API Gateway**: REST API for frontend communication
- **Authentication**: JWT-based user authentication & authorization
- **Job Orchestration**: Manages analysis workflows via BullMQ
- **Real-time Communication**: WebSocket (Socket.io) for progress updates
- **Media Management**: File upload, storage, and metadata extraction
- **PDF Generation**: Creates comprehensive analysis reports
- **Database Access**: Prisma ORM for type-safe database operations

---

## Technology Stack

### Core Technologies

```javascript
// package.json dependencies
{
  "express": "^4.21.2",           // Web framework
  "@prisma/client": "^6.16.1",    // Database ORM
  "socket.io": "^4.8.1",          // WebSocket server
  "bullmq": "^5.58.0",            // Job queue
  "ioredis": "^5.7.0",            // Redis client
  "jsonwebtoken": "^9.0.2",       // JWT authentication
  "bcryptjs": "^3.0.2",           // Password hashing
  "multer": "^2.0.2",             // File upload
  "sharp": "^0.34.3",             // Image processing
  "fluent-ffmpeg": "^2.1.3",      // Video processing
  "axios": "^1.9.0",              // HTTP client (ML Server)
  "zod": "^4.0.17",               // Schema validation
  "winston": "^3.17.0",           // Logging
  "helmet": "^8.1.0",             // Security headers
  "cors": "^2.8.5"                // CORS middleware
}
```

### Development Tools

- **Nodemon**: Auto-restart on file changes
- **Prisma Studio**: Database GUI
- **Jest + Supertest**: Unit & integration testing
- **ESLint**: Code linting
- **Docker**: Containerization

---

## Architecture Layers

### 1. HTTP Layer (`app.js`)

Entry point for all HTTP requests. Configures middleware stack and routes.

```javascript
// Backend/src/app.js
import express from "express";
import cors from "cors";
import helmet from "helmet";
import { apiRateLimiter } from "./middleware/security.middleware.js";
import { errorMiddleware } from "./middleware/error.middleware.js";

const app = express();

// Security & CORS
app.use(helmet());
app.use(cors({ origin: config.FRONTEND_URL, credentials: true }));

// Rate limiting
app.use("/api", apiRateLimiter);

// Body parsing
app.use(express.json({ limit: "20kb" }));
app.use(express.urlencoded({ extended: true, limit: "20kb" }));

// Logging
app.use(morgan("dev", { stream: logger.stream }));

// Routes
app.use("/api/v1/auth", authRoutes);
app.use("/api/v1/media", mediaRoutes);
app.use("/api/v1/monitoring", monitoringRoutes);
app.use("/api/v1/pdf", pdfRoutes);

// Error handling (must be last)
app.use(errorMiddleware);
```

**Key Features:**

- **Security**: Helmet.js for HTTP headers, CORS configuration
- **Rate Limiting**: Prevent abuse with express-rate-limit
- **Request Parsing**: JSON/URL-encoded bodies (20KB limit)
- **Centralized Error Handling**: All errors caught by `errorMiddleware`

### 2. WebSocket Layer (`socket.js`)

Real-time bidirectional communication for progress updates.

```javascript
// Backend/src/config/socket.js
import { Server } from "socket.io";
import { verifyToken } from "../utils/jwt.js";

export const initializeSocketIO = (httpServer) => {
  const io = new Server(httpServer, {
    cors: {
      origin: config.FRONTEND_URL,
      methods: ["GET", "POST"],
      credentials: true,
    },
  });

  // Authentication middleware
  io.use((socket, next) => {
    const token = socket.handshake.auth.token;
    if (!token) {
      return next(new Error("Authentication error: No token provided."));
    }
    try {
      const decoded = verifyToken(token);
      socket.user = decoded; // Attach user to socket
      next();
    } catch (err) {
      return next(new Error("Authentication error: Invalid token."));
    }
  });

  // Connection handler
  io.on("connection", (socket) => {
    logger.info(`User connected: ${socket.user.email}`);
    
    // Join user-specific room for targeted updates
    socket.join(socket.user.userId);
    
    socket.on("disconnect", () => {
      logger.info(`User disconnected: ${socket.user.email}`);
    });
  });

  return io;
};
```

**Key Features:**

- **JWT Authentication**: Validates tokens before connection
- **Room-based Messaging**: Each user has a private room (`userId`)
- **Auto-reconnection**: Client automatically reconnects on disconnect
- **Graceful Degradation**: Fallback to polling if WebSocket unavailable

### 3. Service Layer

Business logic isolated from HTTP/WebSocket layers.

#### Media Service

Orchestrates file upload, storage, and analysis initiation.

```javascript
// Backend/src/services/media.service.js
class MediaService {
  async createAndAnalyzeMedia(file, user, description) {
    // 1. Validate file type
    const mediaType = getMediaType(file.mimetype);
    if (mediaType === "UNKNOWN") {
      throw new ApiError(415, `Unsupported file type: ${file.mimetype}`);
    }

    // 2. Upload to storage (local/Cloudinary)
    const uploadResponse = await storageManager.uploadFile(
      file.path,
      file.originalname,
      mediaType.toLowerCase() + "s"
    );

    // 3. Extract metadata (FFprobe for videos)
    let hasAudio = null;
    let metadata = null;
    if (uploadResponse.metadata) {
      metadata = uploadResponse.metadata;
      if (mediaType === "VIDEO") {
        hasAudio = metadata.audio ? true : false;
      }
    }

    // 4. Create database record
    const mediaRecord = await mediaRepository.create({
      filename: file.originalname,
      url: uploadResponse.url,
      publicId: uploadResponse.publicId,
      mimetype: uploadResponse.mimetype,
      size: uploadResponse.size,
      status: "QUEUED",
      userId: user.id,
      mediaType,
      hasAudio,
      metadata,
    });

    // 5. Queue analysis run (async)
    await this._queueAnalysisRun(mediaRecord, 1); // Run #1

    return mediaRecord;
  }

  async _queueAnalysisRun(media, runNumber) {
    // Create AnalysisRun record
    const run = await mediaRepository.createAnalysisRun(media.id, runNumber);

    // Add job to BullMQ queue
    await mediaFlowProducer.add({
      name: "process-media-flow",
      data: {
        mediaId: media.id,
        runId: run.id,
        userId: media.userId,
      },
    });

    logger.info(`Queued analysis run ${runNumber} for media ${media.id}`);
  }
}
```

#### Model Analysis Service

Communicates with the Python ML Server.

```javascript
// Backend/src/services/modelAnalysis.service.js
class ModelAnalysisService {
  async runAnalysis(mediaPath, modelName, mediaId, userId) {
    // 1. Validate file exists
    if (!fs.existsSync(mediaPath)) {
      throw new ApiError(404, `Media file not found: ${mediaPath}`);
    }

    // 2. Prepare multipart form data
    const form = new FormData();
    form.append("file", fs.createReadStream(mediaPath));
    form.append("media_id", mediaId);
    form.append("user_id", userId);

    // 3. Send to ML Server
    const response = await axios.post(
      `${this.serverUrl}/analyze`,
      form,
      {
        headers: {
          ...form.getHeaders(),
          "X-API-Key": this.apiKey,
        },
        params: { model_name: modelName },
        timeout: this.requestTimeout, // 20 minutes
      }
    );

    // 4. Return structured response
    return {
      success: response.data.success,
      data: response.data.data, // Full JSON payload
      processingTime: response.data.data.processing_time,
    };
  }

  async getServerStatistics() {
    // Cached stats endpoint (60s TTL)
    const cached = await redisCache.get("ml_server_stats");
    if (cached) return JSON.parse(cached);

    const response = await axios.get(`${this.serverUrl}/stats`, {
      headers: { "X-API-Key": this.apiKey },
      timeout: 20000,
    });

    const stats = response.data;
    await redisCache.set("ml_server_stats", JSON.stringify(stats), "EX", 60);
    
    return stats;
  }
}
```

### 4. Repository Layer

Database access abstraction using Prisma.

```javascript
// Backend/src/repositories/media.repository.js
class MediaRepository {
  async create(data) {
    return await prisma.media.create({ data });
  }

  async findById(mediaId) {
    return await prisma.media.findUnique({
      where: { id: mediaId },
      include: {
        user: { select: { id: true, email: true } },
        analysisRuns: {
          include: { analyses: true },
          orderBy: { createdAt: "desc" },
        },
      },
    });
  }

  async createAnalysisRun(mediaId, runNumber) {
    return await prisma.analysisRun.create({
      data: {
        mediaId,
        runNumber,
        status: "QUEUED",
      },
    });
  }

  async updateRunStatus(runId, status) {
    return await prisma.analysisRun.update({
      where: { id: runId },
      data: { status },
    });
  }
}
```

---

## Microservices Communication

### Backend ↔ Server (HTTP)

**Protocol**: REST API over HTTP  
**Authentication**: API Key in `X-API-Key` header  
**Format**: Multipart form-data (file upload) + JSON responses

#### Analysis Request

```javascript
// Backend sends to Server
POST http://server:8000/analyze?model_name=SIGLIP-LSTM-V4
Headers:
  X-API-Key: <server_api_key>
  Content-Type: multipart/form-data

Body:
  file: <binary data>
  media_id: "uuid"
  user_id: "uuid"
```

#### Analysis Response

```json
{
  "success": true,
  "data": {
    "model_name": "SIGLIP-LSTM-V4",
    "prediction": "FAKE",
    "confidence": 0.92,
    "processing_time": 45.2,
    "media_type": "video",
    "metadata": {
      "frames_analyzed": 32,
      "rolling_windows": 8
    }
  }
}
```

### Backend ↔ Frontend (WebSocket)

**Protocol**: Socket.io (WebSocket + fallback to long-polling)  
**Authentication**: JWT token in `socket.handshake.auth.token`  
**Pattern**: Room-based messaging (user-specific rooms)

#### Connection Flow

```javascript
// Frontend connects
const socket = io("http://backend:3000", {
  auth: { token: localStorage.getItem("authToken") }
});

// Backend authenticates & assigns room
socket.join(user.userId); // Private room per user
```

#### Progress Update Event

```javascript
// Backend emits to user's room
io.to(userId).emit("progress_update", {
  mediaId: "uuid",
  event: "model_completed",
  message: "SIGLIP-LSTM-V4 analysis completed",
  data: {
    modelName: "SIGLIP-LSTM-V4",
    prediction: "FAKE",
    confidence: 0.92,
    completedModels: 3,
    totalModels: 5
  },
  timestamp: "2025-10-26T12:34:56Z"
});

// Frontend receives
socket.on("progress_update", (data) => {
  console.log(`Model ${data.data.modelName}: ${data.message}`);
  updateProgressBar(data.data.completedModels / data.data.totalModels);
});
```

### Server → Backend (Redis Pub/Sub)

**Purpose**: ML Server publishes progress events during analysis  
**Channel**: `media-progress-events` (configurable)  
**Pattern**: Publisher-Subscriber (decoupled communication)

#### Event Flow

```text
┌─────────────┐   Pub/Sub    ┌─────────────┐  WebSocket  ┌─────────────┐
│   Server    │─────────────►│   Backend   │────────────►│  Frontend   │
│  (FastAPI)  │   (Redis)    │ (Express.js)│ (Socket.io) │   (React)   │
└─────────────┘              └─────────────┘             └─────────────┘
     │                              │                          │
     │ 1. Analyze video             │                          │
     │ 2. Publish progress          │                          │
     │ ─────────────────────────────►                          │
     │                              │ 3. Convert & emit        │
     │                              │ ─────────────────────────►
     │                              │                          │
     │ 4. Publish completion        │                          │
     │ ─────────────────────────────►                          │
     │                              │ 5. Emit final update     │
     │                              │ ─────────────────────────►
```

#### Redis Event Format (Server → Backend)

```json
{
  "media_id": "uuid",
  "user_id": "uuid",
  "event": "model_started",
  "message": "Starting SIGLIP-LSTM-V4 analysis",
  "data": {
    "model_name": "SIGLIP-LSTM-V4",
    "status": "processing"
  }
}
```

#### Event Service Implementation

```javascript
// Backend/src/services/event.service.js
import { redisSubscriber } from "../config/redis.js";

export function initializeRedisListener(io) {
  // Subscribe to progress channel
  redisSubscriber.subscribe("media-progress-events", (err) => {
    if (err) {
      logger.error("Failed to subscribe to Redis channel:", err);
    } else {
      logger.info("Subscribed to media-progress-events");
    }
  });

  // Handle incoming messages
  redisSubscriber.on("message", (channel, message) => {
    if (channel === "media-progress-events") {
      const progressData = JSON.parse(message);

      // Convert snake_case (Server) to camelCase (Frontend)
      const frontendEvent = {
        mediaId: progressData.media_id,
        userId: progressData.user_id,
        event: progressData.event,
        message: progressData.message,
        data: progressData.data,
        timestamp: new Date().toISOString(),
      };

      // Emit to user's WebSocket room
      io.to(progressData.user_id).emit("progress_update", frontendEvent);
    }
  });

  // Error handling
  redisSubscriber.on("error", (err) => {
    logger.error("Redis subscriber error:", err);
  });
}
```

---

## Request Flow

### Media Upload & Analysis Flow

```text
┌─────────────┐
│  Frontend   │
│   (React)   │
└──────┬──────┘
       │ 1. POST /api/v1/media/upload (multipart/form-data)
       │    Authorization: Bearer <jwt_token>
       │    file: video.mp4
       │    description: "Test video"
       ▼
┌─────────────────────────────────────────────────────────────┐
│                       Backend (Express.js)                   │
├─────────────────────────────────────────────────────────────┤
│  2. Middleware Stack:                                        │
│     - authMiddleware: Verify JWT, extract user              │
│     - multerMiddleware: Parse multipart, save to temp       │
│     - validationMiddleware: Validate file type/size         │
│                                                               │
│  3. mediaController.uploadMedia()                            │
│     ├─ Call mediaService.createAndAnalyzeMedia()            │
│     │  ├─ Upload to storage (local/Cloudinary)              │
│     │  ├─ Extract metadata (FFprobe)                        │
│     │  ├─ Create Media record in PostgreSQL                 │
│     │  └─ Queue AnalysisRun job in BullMQ                   │
│     └─ Return { success: true, media: {...} }               │
│                                                               │
│  4. BullMQ Worker (media.worker.js):                        │
│     ├─ Receive job { mediaId, runId, userId }               │
│     ├─ Create DeepfakeAnalysis records (1 per model)        │
│     ├─ For each active model:                               │
│     │  ├─ Call modelAnalysisService.runAnalysis()           │
│     │  │  └─ POST to Server /analyze                        │
│     │  ├─ Save result to DeepfakeAnalysis                   │
│     │  └─ Update AnalysisRun status                         │
│     └─ Finalize: Update Media status to "ANALYZED"          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ 5. During analysis, Server publishes
                       │    progress events to Redis channel
                       ▼
            ┌─────────────────────┐
            │   Redis Pub/Sub     │
            │  (media-progress)   │
            └─────────┬───────────┘
                      │
                      │ 6. event.service.js listens to Redis
                      │    and emits WebSocket events
                      ▼
            ┌──────────────────────┐
            │ Socket.io Emit       │
            │ to user's room       │
            └─────────┬────────────┘
                      │
                      │ 7. progress_update event
                      │    { mediaId, event, data }
                      ▼
            ┌──────────────────────┐
            │    Frontend (React)  │
            │  Updates UI in       │
            │  real-time           │
            └──────────────────────┘
```

### API Request Flow (General)

```text
HTTP Request
    │
    ├─► Security Middleware (helmet, CORS)
    ├─► Rate Limiter (express-rate-limit)
    ├─► Body Parser (express.json)
    ├─► Logger (morgan)
    │
    ├─► Route Handler (authRoutes, mediaRoutes, etc.)
    │       │
    │       ├─► Authentication Middleware (JWT verification)
    │       ├─► Validation Middleware (express-validator)
    │       │
    │       └─► Controller
    │               │
    │               ├─► Service Layer (business logic)
    │               │       │
    │               │       ├─► Repository (Prisma queries)
    │               │       ├─► External API calls (ML Server)
    │               │       └─► File operations (storage)
    │               │
    │               └─► Response
    │
    └─► Error Middleware (centralized error handling)
```

---

## Design Patterns

### 1. Repository Pattern

**Purpose**: Abstract database access logic  
**Benefits**: Testability, maintainability, swappable data sources

```javascript
// Repository interface
class MediaRepository {
  async create(data) { /* Prisma logic */ }
  async findById(id) { /* Prisma logic */ }
  async update(id, data) { /* Prisma logic */ }
  async delete(id) { /* Prisma logic */ }
}

// Usage in service (no direct Prisma calls)
class MediaService {
  async getMedia(id) {
    return await mediaRepository.findById(id);
  }
}
```

### 2. Service Layer Pattern

**Purpose**: Encapsulate business logic  
**Benefits**: Reusable across controllers, testable in isolation

```javascript
// Services handle complex workflows
class MediaService {
  async createAndAnalyzeMedia(file, user, description) {
    // Multi-step workflow:
    // 1. Validate file
    // 2. Upload to storage
    // 3. Create database record
    // 4. Queue analysis job
  }
}
```

### 3. Factory Pattern

**Purpose**: Create objects without exposing instantiation logic  
**Used in**: Storage Manager (local vs Cloudinary)

```javascript
// Backend/src/storage/storage.manager.js
class StorageManager {
  constructor() {
    const provider = config.STORAGE_PROVIDER; // "local" or "cloudinary"
    
    if (provider === "cloudinary") {
      this.strategy = new CloudinaryStorage();
    } else {
      this.strategy = new LocalStorage();
    }
  }

  async uploadFile(filePath, filename, folder) {
    return await this.strategy.uploadFile(filePath, filename, folder);
  }
}
```

### 4. Middleware Chain Pattern

**Purpose**: Modular request processing  
**Benefits**: Separation of concerns, reusable middleware

```javascript
app.use("/api/v1/media", [
  authMiddleware,          // 1. Verify JWT
  upload.single("file"),   // 2. Parse file upload
  validateFileType,        // 3. Validate MIME type
  mediaController.upload   // 4. Business logic
]);
```

### 5. Pub/Sub Pattern

**Purpose**: Decouple event producers from consumers  
**Used in**: Redis progress events (Server → Backend → Frontend)

```javascript
// Publisher (ML Server)
await redisPublisher.publish("media-progress-events", JSON.stringify(event));

// Subscriber (Backend)
redisSubscriber.on("message", (channel, message) => {
  const event = JSON.parse(message);
  io.to(event.user_id).emit("progress_update", event);
});
```

### 6. Queue/Worker Pattern

**Purpose**: Asynchronous job processing  
**Used in**: BullMQ for analysis workflows

```javascript
// Producer (adds job to queue)
await mediaFlowProducer.add({
  name: "process-media-flow",
  data: { mediaId, runId, userId }
});

// Consumer (processes job in background)
mediaWorker.on("active", async (job) => {
  const { mediaId, runId } = job.data;
  await runAnalysisWorkflow(mediaId, runId);
});
```

---

## Service Configuration

### Environment Variables (`.env`)

```bash
# Server
NODE_ENV=production
PORT=3000
API_BASE_URL=http://localhost:3000

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/drishtiksha

# Redis
REDIS_URL=redis://localhost:6379

# Queue
MEDIA_PROCESSING_QUEUE_NAME=media-processing-queue
MEDIA_PROGRESS_CHANNEL_NAME=media-progress-events

# Security
JWT_SECRET=your-256-bit-secret
JWT_EXPIRES_IN=7d
BCRYPT_ROUNDS=12

# CORS
FRONTEND_URL=http://localhost:5173

# Storage
STORAGE_PROVIDER=local # or "cloudinary"
LOCAL_STORAGE_PATH=public/media
ASSETS_BASE_URL=http://localhost:3001

# Cloudinary (if STORAGE_PROVIDER=cloudinary)
CLOUDINARY_CLOUD_NAME=your-cloud
CLOUDINARY_API_KEY=your-key
CLOUDINARY_API_SECRET=your-secret

# ML Server
SERVER_URL=http://localhost:8000
SERVER_API_KEY=your-server-api-key
```

### Configuration Validation (Zod)

```javascript
// Backend/src/config/env.js
import { z } from "zod";

const envSchema = z.object({
  NODE_ENV: z.enum(["development", "production", "test"]).default("development"),
  PORT: z.coerce.number().default(3000),
  DATABASE_URL: z.string().url(),
  REDIS_URL: z.string().url(),
  JWT_SECRET: z.string().min(1),
  STORAGE_PROVIDER: z.enum(["local", "cloudinary"]).default("local"),
  SERVER_URL: z.string().url(),
  SERVER_API_KEY: z.string().min(1),
}).superRefine((data, ctx) => {
  // Conditional validation: Cloudinary keys required if provider is cloudinary
  if (data.STORAGE_PROVIDER === "cloudinary") {
    if (!data.CLOUDINARY_CLOUD_NAME) {
      ctx.addIssue({
        code: z.custom,
        path: ["CLOUDINARY_CLOUD_NAME"],
        message: "CLOUDINARY_CLOUD_NAME required when STORAGE_PROVIDER=cloudinary"
      });
    }
  }
});

// Validate on startup
const validatedEnv = envSchema.safeParse(process.env);
if (!validatedEnv.success) {
  console.error("❌ Invalid environment variables:");
  console.error(validatedEnv.error.flatten().fieldErrors);
  process.exit(1);
}

export const config = validatedEnv.data;
```

### Service Initialization

```javascript
// Backend/server.js
import { createServer } from "http";
import { app } from "./src/app.js";
import { initializeSocketIO } from "./src/config/socket.js";
import { initializeQueueEvents } from "./src/workers/queueEvents.js";
import { initializeRedisListener } from "./src/services/event.service.js";
import { connectServices, disconnectServices } from "./src/config/index.js";

const startServer = async () => {
  const httpServer = createServer(app);
  
  // 1. Initialize Socket.io
  const io = initializeSocketIO(httpServer);
  app.set("io", io); // Make io available in routes
  
  // 2. Initialize BullMQ queue event listeners
  initializeQueueEvents(io);
  
  // 3. Initialize Redis pub/sub listener
  initializeRedisListener(io);
  
  // 4. Connect to databases (Prisma + Redis)
  await connectServices();
  
  // 5. Start HTTP server
  httpServer.listen(config.PORT, () => {
    logger.info(`🚀 Server running at http://localhost:${config.PORT}`);
  });
};

startServer();
```

---

## Background Workers

### BullMQ Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                      BullMQ Queue System                      │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────┐      ┌──────────────┐      ┌─────────────┐ │
│  │  Producer   │─────►│     Queue    │─────►│   Worker    │ │
│  │ (API Route) │      │    (Redis)   │      │ (Processor) │ │
│  └─────────────┘      └──────────────┘      └─────────────┘ │
│         │                     │                      │        │
│         │                     │                      │        │
│         │ Add job             │ Store job            │ Process│
│         │                     │                      │        │
│         ▼                     ▼                      ▼        │
│  { mediaId, runId }    Redis List + Hash     Run ML models   │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

### Media Worker

```javascript
// Backend/src/workers/media.worker.js
import { Worker } from "bullmq";
import { config, redisConnectionOptionsForBullMQ, prisma } from "../config/index.js";
import { modelAnalysisService } from "../services/modelAnalysis.service.js";

const mediaWorker = new Worker(
  config.MEDIA_PROCESSING_QUEUE_NAME,
  async (job) => {
    const { mediaId, runId, userId } = job.data;
    
    logger.info(`[Worker] Processing job ${job.id} for media ${mediaId}`);
    
    // 1. Get active models from ML Server
    const serverStats = await modelAnalysisService.getServerStatistics();
    const activeModels = serverStats.active_models;
    
    // 2. Create DeepfakeAnalysis records (1 per model)
    const analysisPromises = activeModels.map(modelName =>
      prisma.deepfakeAnalysis.create({
        data: {
          analysisRunId: runId,
          modelName,
          status: "PENDING",
          prediction: "",
          confidence: 0,
          resultPayload: {},
        }
      })
    );
    await Promise.all(analysisPromises);
    
    // 3. Update run status to PROCESSING
    await prisma.analysisRun.update({
      where: { id: runId },
      data: { status: "PROCESSING" }
    });
    
    // 4. Run each model analysis
    const media = await prisma.media.findUnique({ where: { id: mediaId } });
    const mediaPath = getLocalPath(media.publicId);
    
    for (const modelName of activeModels) {
      try {
        // Call ML Server
        const result = await modelAnalysisService.runAnalysis(
          mediaPath,
          modelName,
          mediaId,
          userId
        );
        
        // Save result
        await prisma.deepfakeAnalysis.updateMany({
          where: { analysisRunId: runId, modelName },
          data: {
            status: "COMPLETED",
            prediction: result.data.prediction,
            confidence: result.data.confidence,
            processingTime: result.data.processing_time,
            mediaType: result.data.media_type,
            resultPayload: result.data,
          }
        });
        
        logger.info(`[Worker] ${modelName} completed for run ${runId}`);
      } catch (error) {
        logger.error(`[Worker] ${modelName} failed for run ${runId}:`, error);
        
        await prisma.deepfakeAnalysis.updateMany({
          where: { analysisRunId: runId, modelName },
          data: {
            status: "FAILED",
            errorMessage: error.message,
          }
        });
      }
    }
    
    // 5. Finalize run
    const allAnalyses = await prisma.deepfakeAnalysis.findMany({
      where: { analysisRunId: runId }
    });
    
    const allCompleted = allAnalyses.every(a => a.status === "COMPLETED");
    const anyCompleted = allAnalyses.some(a => a.status === "COMPLETED");
    const finalStatus = allCompleted ? "ANALYZED" : (anyCompleted ? "ANALYZED" : "FAILED");
    
    await prisma.analysisRun.update({
      where: { id: runId },
      data: { status: finalStatus }
    });
    
    await prisma.media.update({
      where: { id: mediaId },
      data: { 
        status: finalStatus,
        latestAnalysisRunId: runId 
      }
    });
    
    logger.info(`[Worker] Job ${job.id} completed with status ${finalStatus}`);
    
    return { mediaId, runId, status: finalStatus };
  },
  {
    connection: redisConnectionOptionsForBullMQ,
    concurrency: 2, // Process 2 jobs simultaneously
    limiter: {
      max: 5,      // Max 5 jobs
      duration: 60000 // per 60 seconds
    }
  }
);

// Event listeners
mediaWorker.on("completed", (job) => {
  logger.info(`Job ${job.id} completed successfully`);
});

mediaWorker.on("failed", (job, err) => {
  logger.error(`Job ${job.id} failed: ${err.message}`);
});
```

### Queue Events (Real-time Progress)

```javascript
// Backend/src/workers/queueEvents.js
import { QueueEvents } from "bullmq";

export function initializeQueueEvents(io) {
  const queueEvents = new QueueEvents(config.MEDIA_PROCESSING_QUEUE_NAME, {
    connection: redisConnectionOptionsForBullMQ,
  });

  // Job completed
  queueEvents.on("completed", async ({ jobId, returnvalue }) => {
    const { mediaId, runId, status } = returnvalue;
    
    // Fetch updated media record
    const media = await mediaRepository.findById(mediaId);
    
    // Emit to user's WebSocket room
    if (media?.user) {
      io.to(media.user.id).emit("media_update", media);
      logger.info(`Emitted 'media_update' for media ${mediaId} to user ${media.user.id}`);
    }
  });

  // Job failed
  queueEvents.on("failed", async ({ jobId, failedReason }) => {
    logger.error(`Job ${jobId} failed: ${failedReason}`);
    
    // Extract runId from jobId (format: runId-modelName)
    const runId = jobId.split("-")[0];
    
    const run = await prisma.analysisRun.findUnique({
      where: { id: runId },
      include: { media: true }
    });
    
    if (run?.media) {
      // Emit error to user
      io.to(run.media.userId).emit("processing_error", {
        mediaId: run.media.id,
        filename: run.media.filename,
        error: failedReason,
      });
    }
  });

  logger.info("🎧 BullMQ QueueEvents listener initialized");
}
```

---

## Error Handling & Resilience

### Custom Error Class

```javascript
// Backend/src/utils/ApiError.js
export class ApiError extends Error {
  constructor(statusCode, message, errors = null) {
    super(message);
    this.statusCode = statusCode;
    this.errors = errors;
    this.isOperational = true; // Distinguishes from programming errors
    Error.captureStackTrace(this, this.constructor);
  }
}
```

### Error Middleware

```javascript
// Backend/src/middleware/error.middleware.js
export const errorMiddleware = (err, req, res, next) => {
  let { statusCode, message, errors } = err;

  // Default to 500 if not set
  if (!statusCode) statusCode = 500;
  if (!message) message = "Internal Server Error";

  // Log error (except 4xx client errors)
  if (statusCode >= 500) {
    logger.error(`[${req.method}] ${req.path} >> StatusCode: ${statusCode}, Message: ${message}`);
    logger.error(err.stack);
  }

  // Send JSON response
  res.status(statusCode).json({
    success: false,
    message,
    errors,
    ...(config.NODE_ENV === "development" && { stack: err.stack })
  });
};
```

### Graceful Shutdown

```javascript
// Backend/server.js
const shutdown = async (signal) => {
  logger.info(`${signal} received. Shutting down gracefully...`);
  
  httpServer.close(async () => {
    logger.info("HTTP server closed");
    
    // Close database connections
    await disconnectServices();
    
    logger.info("All connections closed. Shutdown complete.");
    process.exit(0);
  });

  // Force shutdown after 10s
  setTimeout(() => {
    logger.error("Could not close connections in time. Forcing shutdown.");
    process.exit(1);
  }, 10000);
};

process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT", () => shutdown("SIGINT"));
```

### Stuck Job Detection

```javascript
// Backend/src/scripts/check-stuck-runs.js
export async function checkAndFinalizeStuckRuns() {
  // Find runs stuck in PROCESSING for >30 minutes
  const stuckRuns = await prisma.analysisRun.findMany({
    where: {
      status: "PROCESSING",
      createdAt: {
        lt: new Date(Date.now() - 30 * 60 * 1000) // 30 minutes ago
      }
    },
    include: { analyses: true }
  });

  for (const run of stuckRuns) {
    const completed = run.analyses.filter(a => a.status === "COMPLETED").length;
    const total = run.analyses.length;
    
    if (completed > 0) {
      // Partial completion - mark as ANALYZED
      await prisma.analysisRun.update({
        where: { id: run.id },
        data: { status: "ANALYZED" }
      });
      logger.info(`Finalized stuck run ${run.id}: ${completed}/${total} models completed`);
    } else {
      // No completions - mark as FAILED
      await prisma.analysisRun.update({
        where: { id: run.id },
        data: { status: "FAILED" }
      });
      logger.warn(`Failed stuck run ${run.id}: 0/${total} models completed`);
    }
  }
}

// Run every 30 seconds
setInterval(checkAndFinalizeStuckRuns, 30000);
```

---

## Deployment Architecture

### Docker Compose Setup

```yaml
# Backend/docker-compose.yml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "3000:3000"
    environment:
      NODE_ENV: production
      DATABASE_URL: postgresql://user:pass@postgres:5432/drishtiksha
      REDIS_URL: redis://redis:6379
      SERVER_URL: http://server:8000
      FRONTEND_URL: http://localhost:5173
    depends_on:
      - postgres
      - redis
      - server
    volumes:
      - ./public/media:/app/public/media
    restart: unless-stopped

  worker:
    build: .
    command: npm run worker
    environment:
      NODE_ENV: production
      DATABASE_URL: postgresql://user:pass@postgres:5432/drishtiksha
      REDIS_URL: redis://redis:6379
      SERVER_URL: http://server:8000
    depends_on:
      - postgres
      - redis
      - server
    restart: unless-stopped

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: drishtiksha
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### Production Considerations

1. **Reverse Proxy**: Use Nginx/Traefik for SSL termination and load balancing
2. **Process Manager**: PM2 for multiple instances and auto-restart
3. **Monitoring**: Prometheus + Grafana for metrics
4. **Logging**: Winston → Elasticsearch/Loki for centralized logging
5. **Secrets Management**: Use Docker Secrets or Vault for sensitive env vars
6. **Database Backups**: Automated PostgreSQL backups with point-in-time recovery
7. **Redis Persistence**: Enable AOF (Append-Only File) for durability

### PM2 Ecosystem File

```javascript
// ecosystem.config.js
module.exports = {
  apps: [
    {
      name: "backend",
      script: "server.js",
      instances: 4,
      exec_mode: "cluster",
      env: {
        NODE_ENV: "production",
        PORT: 3000
      }
    },
    {
      name: "worker",
      script: "src/workers/media.worker.js",
      instances: 2,
      exec_mode: "fork",
      env: {
        NODE_ENV: "production"
      }
    }
  ]
};
```

---

## Summary

The Drishtiksha Backend is a **production-ready Node.js microservice** that:

✅ **Orchestrates** ML analysis workflows via BullMQ  
✅ **Authenticates** users with JWT-based auth  
✅ **Communicates** with ML Server via REST API and Redis Pub/Sub  
✅ **Streams** real-time progress via Socket.io WebSockets  
✅ **Persists** data in PostgreSQL with Prisma ORM  
✅ **Scales** horizontally with clustered workers  
✅ **Handles** errors gracefully with retry logic and stuck job detection  
✅ **Deploys** easily with Docker Compose

**Next Steps:**

- [API Routes & Endpoints Documentation](./API-Routes.md)
- [Database Schema & Prisma Documentation](./Database-Schema.md)
- [Services & Business Logic Documentation](./Services.md)
- [Middleware & Authentication Documentation](./Middleware.md)
- [WebSocket & Real-time Updates Documentation](./WebSocket.md)
