# Database Schema & Prisma ORM

Complete documentation for the **Drishtiksha Backend** database schema, Prisma ORM integration, repository patterns, migrations strategy, and query optimization techniques.

---

## Table of Contents

1. [Overview](#overview)
2. [Database Stack](#database-stack)
3. [Schema Architecture](#schema-architecture)
4. [Core Models](#core-models)
5. [Analysis & Versioning Models](#analysis--versioning-models)
6. [Monitoring Models](#monitoring-models)
7. [Relationships & Cascades](#relationships--cascades)
8. [Indexes & Performance](#indexes--performance)
9. [Enums](#enums)
10. [Repository Pattern](#repository-pattern)
11. [Migrations Strategy](#migrations-strategy)
12. [Query Patterns](#query-patterns)
13. [Analytics Queries](#analytics-queries)
14. [Transaction Handling](#transaction-handling)
15. [Best Practices](#best-practices)

---

## Overview

### Database Philosophy

The Drishtiksha database schema follows a **versioned analysis** architecture where:

- Each media file can have **multiple analysis runs** over time
- Each run contains **multiple model analyses** (one per active model)
- Results are stored as **flexible JSON payloads** with **promoted fields** for efficient querying
- **Cascade deletes** ensure referential integrity

### Key Features

✅ **Type-Safe**: Prisma generates TypeScript-safe database clients  
✅ **Versioned Analysis**: Track multiple analysis runs per media item  
✅ **Flexible Schema**: JSON columns for dynamic ML model responses  
✅ **Optimized Queries**: Strategic indexes for common access patterns  
✅ **Atomic Operations**: Transaction support for complex workflows  
✅ **Automatic Timestamps**: `createdAt` and `updatedAt` tracking  
✅ **Cascade Deletes**: Automatic cleanup of related records

---

## Database Stack

### Technology

```yaml
Database: PostgreSQL 16
ORM: Prisma 6.16.1
Client: @prisma/client
Migration Tool: Prisma Migrate
Studio: Prisma Studio (GUI)
```

### Connection Configuration

```javascript
// Backend/src/config/index.js
import { PrismaClient } from '@prisma/client';

export const prisma = new PrismaClient({
  log: config.NODE_ENV === 'development' 
    ? ['warn', 'error']   // Development: log warnings and errors
    : ['error'],           // Production: only errors
});

// Connection string from environment
// DATABASE_URL=postgresql://user:password@localhost:5432/drishtiksha
```

### Service Lifecycle

```javascript
// Connect on startup
export const connectServices = async () => {
  await prisma.$connect();
  logger.info('🗄️ Database connected successfully.');
};

// Disconnect on shutdown
export const disconnectServices = async () => {
  await prisma.$disconnect();
  logger.info('🔌 Database connection closed.');
};
```

---

## Schema Architecture

### Entity Relationship Diagram

```text
┌─────────────┐
│    User     │
│             │
│ - id        │
│ - email     │◄──────────────────────────────┐
│ - password  │                                │
│ - firstName │                                │
│ - lastName  │                                │
│ - role      │                                │
└─────────────┘                                │
                                                │ userId (FK)
                                                │ CASCADE DELETE
┌─────────────────────────────────────────────┼─────────────┐
│                    Media                     │             │
│                                              │             │
│ - id                                         │             │
│ - filename                                   │             │
│ - url                                        │             │
│ - publicId                                   │             │
│ - mimetype                                   │             │
│ - size                                       │             │
│ - description                                │             │
│ - status (QUEUED|PROCESSING|ANALYZED|FAILED) │             │
│ - mediaType (VIDEO|IMAGE|AUDIO)              │             │
│ - hasAudio (for videos)                      │             │
│ - metadata (JSON: FFprobe data)              │             │
│ - latestAnalysisRunId                        │             │
└──────────────────┬───────────────────────────┴─────────────┘
                   │
                   │ mediaId (FK)
                   │ CASCADE DELETE
                   │
┌──────────────────▼───────────────────────────────────────────┐
│                      AnalysisRun                              │
│                                                                │
│ - id                                                           │
│ - mediaId                                                      │
│ - runNumber (sequential: 1, 2, 3...)                          │
│ - status (QUEUED|PROCESSING|ANALYZED|FAILED)                  │
│ - createdAt                                                    │
│                                                                │
│ UNIQUE (mediaId, runNumber) ◄── Ensures sequential runs      │
└──────────────────┬─────────────────────────────────────────────┘
                   │
                   │ analysisRunId (FK)
                   │ CASCADE DELETE
                   │
┌──────────────────▼──────────────────────────────────────────┐
│                  DeepfakeAnalysis                            │
│                                                               │
│ - id                                                          │
│ - analysisRunId                                               │
│ - modelName (SIGLIP-LSTM-V4, etc.)                           │
│ - prediction (REAL|FAKE)                                     │
│ - confidence (0.0 - 1.0)                                     │
│ - status (PENDING|COMPLETED|FAILED)                          │
│ - errorMessage                                                │
│                                                               │
│ ✨ Promoted Fields (extracted from resultPayload):           │
│ - processingTime (seconds)                                   │
│ - mediaType (video|image|audio)                              │
│                                                               │
│ - resultPayload (JSON: full ML Server response)             │
│   {                                                           │
│     "model_name": "SIGLIP-LSTM-V4",                          │
│     "prediction": "FAKE",                                    │
│     "confidence": 0.92,                                      │
│     "processing_time": 45.2,                                 │
│     "metadata": { ... }                                      │
│   }                                                           │
│                                                               │
│ INDEXES: analysisRunId, modelName, processingTime,          │
│          mediaType, confidence                               │
└───────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────┐
│                      ServerHealth                             │
│                        (Monitoring)                           │
│                                                                │
│ - id                                                           │
│ - status (HEALTHY|UNHEALTHY|DEGRADED|UNKNOWN)                │
│ - responseTimeMs                                               │
│ - statsPayload (JSON: full /stats response)                  │
│ - checkedAt                                                    │
└────────────────────────────────────────────────────────────────┘
```

---

## Core Models

### User Model

Represents authenticated users of the platform.

```prisma
model User {
  id        String   @id @default(uuid())
  email     String   @unique
  firstName String
  lastName  String
  password  String   // Bcrypt hashed (12 rounds)
  role      Role     @default(USER)   // USER | ADMIN
  isActive  Boolean  @default(true)   // Soft delete flag
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt

  media Media[]  // One-to-many relationship

  @@map("users")
}
```

**Fields**:

- `id`: UUID primary key
- `email`: Unique email address (used for login)
- `password`: Bcrypt-hashed password (rounds: 12)
- `role`: Enum (`USER` | `ADMIN`) for authorization
- `isActive`: Soft delete flag (inactive users can't login)
- `media`: Relationship to all uploaded media files

**Constraints**:

- `email` must be unique
- `password` never exposed in API responses (excluded via repository pattern)

---

### Media Model

Represents uploaded media files (video, audio, image).

```prisma
model Media {
  id          String      @id @default(uuid())
  filename    String      // Original filename
  url         String      // Public access URL
  publicId    String      // Storage provider ID (local path or Cloudinary ID)
  mimetype    String      // MIME type (video/mp4, image/jpeg, etc.)
  size        Int         // File size in bytes
  description String?     // Optional user description
  status      MediaStatus @default(QUEUED)  // Latest run status
  mediaType   MediaType   // VIDEO | IMAGE | AUDIO | UNKNOWN
  hasAudio    Boolean?    @map("has_audio")  // Video-only: audio track exists

  // FFprobe metadata (JSON)
  metadata Json?  // { format: {...}, video: {...}, audio: {...} }

  userId    String
  user      User     @relation(fields: [userId], references: [id], onDelete: Cascade)
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt

  analysisRuns AnalysisRun[]  // One-to-many: multiple analysis runs

  latestAnalysisRunId String?  // Performance optimization

  @@index([userId, createdAt(sort: Desc)])
  @@map("media")
}
```

**Fields**:

- `publicId`:
  - **Local storage**: Relative path like `videos/abc123-uuid.mp4`
  - **Cloudinary**: Cloudinary public ID
- `status`: Reflects the status of the **latest** analysis run
- `mediaType`: Determined from MIME type via `getMediaType()` utility
- `hasAudio`:
  - `true`: Video has audio track
  - `false`: Video has no audio (silent)
  - `null`: Not a video (image/audio file)
- `metadata`: Rich JSON extracted by FFprobe:

```json
{
  "format": {
    "duration": 30.5,
    "bit_rate": 4128000,
    "size": 15728640
  },
  "video": {
    "codec": "h264",
    "width": 1920,
    "height": 1080,
    "fps": 30,
    "bitrate": 3800000
  },
  "audio": {
    "codec": "aac",
    "sample_rate": 48000,
    "channels": 2,
    "bitrate": 128000
  }
}
```

**Relationships**:

- `user`: Many-to-one (Media → User), cascade delete
- `analysisRuns`: One-to-many (Media → AnalysisRun), cascade delete

**Indexes**:

- `[userId, createdAt DESC]`: Optimize fetching user's media sorted by newest first

---

## Analysis & Versioning Models

### AnalysisRun Model

Represents a single execution of the multi-model analysis pipeline. Users can re-run analysis multiple times.

```prisma
model AnalysisRun {
  id      String @id @default(uuid())
  mediaId String
  media   Media  @relation(fields: [mediaId], references: [id], onDelete: Cascade)

  runNumber Int  // Sequential: 1, 2, 3...

  status MediaStatus @default(QUEUED)  // Run-level status

  createdAt DateTime @default(now())

  analyses DeepfakeAnalysis[]  // One-to-many: results per model

  @@unique([mediaId, runNumber])  // Ensures sequential runs
  @@map("analysis_runs")
}
```

**Fields**:

- `runNumber`: Auto-incremented per media item
  - First analysis: `runNumber = 1`
  - Re-run: `runNumber = 2`
  - Second re-run: `runNumber = 3`
- `status`: Aggregated status from all child analyses
  - `QUEUED`: Just created, not started
  - `PROCESSING`: At least one model is processing
  - `ANALYZED`: All models completed (or partial success)
  - `FAILED`: All models failed

**Unique Constraint**:

```sql
UNIQUE (mediaId, runNumber)
```

Prevents duplicate run numbers for the same media item.

**Cascade Delete**:

When Media is deleted → All AnalysisRuns deleted → All DeepfakeAnalyses deleted

---

### DeepfakeAnalysis Model

Represents the result from a **single model** for a given AnalysisRun.

```prisma
model DeepfakeAnalysis {
  id            String      @id @default(uuid())
  analysisRunId String
  analysisRun   AnalysisRun @relation(fields: [analysisRunId], references: [id], onDelete: Cascade)

  modelName    String         // e.g., "SIGLIP-LSTM-V4"
  prediction   String         // "REAL" | "FAKE"
  confidence   Float          // 0.0 - 1.0
  status       AnalysisStatus @default(PENDING)
  errorMessage String?        // Set if status = FAILED

  // ✨ Promoted fields (extracted from resultPayload for efficient querying)
  processingTime Float?  @map("processing_time")  // Seconds
  mediaType      String? @map("media_type")       // "video" | "image" | "audio"

  // Full ML Server response (flexible JSON)
  resultPayload Json @map("result_payload")

  createdAt DateTime @default(now()) @map("created_at")
  updatedAt DateTime @updatedAt @map("updated_at")

  @@index([analysisRunId])
  @@index([modelName])
  @@index([processingTime])
  @@index([mediaType])
  @@index([confidence])
  @@map("deepfake_analyses")
}
```

**Promoted Fields Strategy**:

Instead of hardcoding every possible field from ML Server responses, we:

1. **Store full response** in `resultPayload` (JSON)
2. **Promote key fields** for database-level filtering/sorting:
   - `processingTime`: Enable queries like "fastest analyses"
   - `mediaType`: Filter by media type at database level
   - `confidence`: Sort by confidence scores

**Example `resultPayload`**:

```json
{
  "model_name": "SIGLIP-LSTM-V4",
  "prediction": "FAKE",
  "confidence": 0.92,
  "processing_time": 45.2,
  "media_type": "video",
  "metadata": {
    "frames_analyzed": 32,
    "rolling_windows": 8,
    "temporal_consistency": 0.88
  },
  "visualization": {
    "attention_map_url": "http://..."
  }
}
```

**Indexes**:

- `analysisRunId`: Fast lookup of all analyses in a run
- `modelName`: Filter by specific model
- `processingTime`: Sort by performance
- `mediaType`: Filter by media type
- `confidence`: Sort by confidence

---

## Monitoring Models

### ServerHealth Model

Tracks ML Server health over time for monitoring dashboards.

```prisma
model ServerHealth {
  id             String @id @default(uuid())
  status         String  // "HEALTHY" | "UNHEALTHY" | "DEGRADED" | "UNKNOWN"
  responseTimeMs Int     // Server response time in milliseconds

  // Full /stats response from ML Server
  statsPayload Json

  checkedAt DateTime @default(now())

  @@map("server_health")
}
```

**Example `statsPayload`**:

```json
{
  "status": "HEALTHY",
  "uptime": 86400,
  "active_models": ["SIGLIP-LSTM-V4", "EFFICIENTNET-B7-V1"],
  "total_models": 15,
  "device": "cuda",
  "memory_usage": {
    "gpu": {
      "allocated": 4096,
      "reserved": 8192,
      "total": 16384
    }
  },
  "responseTimeMs": 120
}
```

**Use Cases**:

- Historical health tracking
- Performance trend analysis
- Uptime monitoring
- Alerting on degradation

---

## Relationships & Cascades

### Cascade Delete Chains

```text
User
  │ ON DELETE CASCADE
  └─► Media
        │ ON DELETE CASCADE
        ├─► AnalysisRun
        │     │ ON DELETE CASCADE
        │     └─► DeepfakeAnalysis
        └─► (All cascaded automatically)
```

**Deletion Flow**:

```javascript
// Delete user
await prisma.user.delete({ where: { id: userId } });

// Automatically deletes:
// - All Media records (userId FK)
// - All AnalysisRun records (mediaId FK)
// - All DeepfakeAnalysis records (analysisRunId FK)
```

**Benefits**:

- **Data Integrity**: No orphaned records
- **Automatic Cleanup**: No manual cascade logic needed
- **Referential Integrity**: PostgreSQL enforces constraints

---

## Indexes & Performance

### Strategic Indexes

#### 1. Media User Lookup

```prisma
@@index([userId, createdAt(sort: Desc)])
```

**Optimizes**:

```sql
SELECT * FROM media 
WHERE userId = ? 
ORDER BY createdAt DESC;
```

**Use Case**: Fetching user's media sorted by newest first

---

#### 2. DeepfakeAnalysis Queries

```prisma
@@index([analysisRunId])       // Fetch all analyses for a run
@@index([modelName])            // Filter by specific model
@@index([processingTime])       // Sort by performance
@@index([mediaType])            // Filter by media type
@@index([confidence])           // Sort by confidence
```

**Optimizes**:

```sql
-- Get all analyses for a run
SELECT * FROM deepfake_analyses WHERE analysisRunId = ?;

-- Find fastest analyses
SELECT * FROM deepfake_analyses 
WHERE processingTime IS NOT NULL 
ORDER BY processingTime ASC;

-- Get video analyses only
SELECT * FROM deepfake_analyses WHERE mediaType = 'video';

-- High-confidence predictions
SELECT * FROM deepfake_analyses 
WHERE confidence > 0.9 
ORDER BY confidence DESC;
```

---

#### 3. Composite Index Performance

PostgreSQL uses the composite index `[userId, createdAt DESC]` for:

✅ `WHERE userId = ?`  
✅ `WHERE userId = ? ORDER BY createdAt DESC`  
❌ `WHERE createdAt > ?` (needs separate index)

---

## Enums

### Role Enum

```prisma
enum Role {
  USER
  ADMIN
}
```

**Usage**: User authorization levels

---

### MediaStatus Enum

```prisma
enum MediaStatus {
  QUEUED      // Waiting to be processed
  PROCESSING  // Currently analyzing
  ANALYZED    // Analysis complete
  FAILED      // Analysis failed
}
```

**Usage**: Track status of Media and AnalysisRun

**State Transitions**:

```text
QUEUED → PROCESSING → ANALYZED
   │                      ↓
   └───────────────► FAILED
```

---

### AnalysisStatus Enum

```prisma
enum AnalysisStatus {
  PENDING    // Not started yet
  COMPLETED  // Successfully finished
  FAILED     // Errored during analysis
}
```

**Usage**: Track individual model analysis status

---

### MediaType Enum

```prisma
enum MediaType {
  VIDEO
  IMAGE
  AUDIO
  UNKNOWN  // Fallback for unsupported types
}
```

**Determined by**:

```javascript
// Backend/src/utils/media.js
export function getMediaType(mimetype) {
  if (mimetype.startsWith('video/')) return 'VIDEO';
  if (mimetype.startsWith('image/')) return 'IMAGE';
  if (mimetype.startsWith('audio/')) return 'AUDIO';
  return 'UNKNOWN';
}
```

---

## Repository Pattern

### Architecture

The repository pattern abstracts database access from business logic.

```text
┌──────────────┐
│  Controller  │
└──────┬───────┘
       │ Calls
       ▼
┌──────────────┐
│   Service    │
└──────┬───────┘
       │ Calls
       ▼
┌──────────────┐
│  Repository  │  ◄── Abstracts Prisma queries
└──────┬───────┘
       │ Uses
       ▼
┌──────────────┐
│    Prisma    │
└──────────────┘
```

---

### Media Repository

**File**: `Backend/src/repositories/media.repository.js`

#### Include Patterns

```javascript
// Latest run only (for list views)
const mediaWithLatestRunDetails = {
  include: {
    user: {
      select: { id: true, firstName: true, lastName: true, email: true },
    },
    analysisRuns: {
      orderBy: { runNumber: 'desc' },
      take: 1,  // Only latest run
      include: {
        analyses: {
          orderBy: { createdAt: 'asc' },
        },
      },
    },
  },
};

// All runs (for detail views)
const mediaWithAllRunsDetails = {
  include: {
    user: { /* ... */ },
    analysisRuns: {
      orderBy: { runNumber: 'desc' },
      // No take limit - include all runs
      include: {
        analyses: { /* ... */ },
      },
    },
  },
};
```

---

#### CRUD Operations

```javascript
export const mediaRepository = {
  // Create media
  async create(mediaData) {
    return prisma.media.create({ data: mediaData });
  },

  // Find by ID with all runs
  async findById(mediaId) {
    return prisma.media.findUnique({
      where: { id: mediaId },
      ...mediaWithAllRunsDetails,
    });
  },

  // Find by ID and verify ownership
  async findByIdAndUserId(mediaId, userId) {
    return prisma.media.findFirst({
      where: { id: mediaId, userId },
      ...mediaWithAllRunsDetails,
    });
  },

  // Get all user's media (latest run only)
  async findAllByUserId(userId) {
    return prisma.media.findMany({
      where: { userId },
      ...mediaWithLatestRunDetails,
      orderBy: { createdAt: 'desc' },
    });
  },

  // Update media
  async update(mediaId, updateData) {
    return prisma.media.update({
      where: { id: mediaId },
      data: updateData,
    });
  },

  // Delete media (cascades to runs & analyses)
  async deleteById(mediaId) {
    return prisma.media.delete({ where: { id: mediaId } });
  },
};
```

---

#### Analysis Run Operations

```javascript
// Create new analysis run
async createAnalysisRun(mediaId, runNumber) {
  return prisma.analysisRun.create({
    data: { mediaId, runNumber, status: 'QUEUED' },
  });
},

// Update run status
async updateRunStatus(runId, status) {
  return prisma.analysisRun.update({
    where: { id: runId },
    data: { status },
  });
},

// Get latest run number (for sequential numbering)
async findLatestRunNumber(mediaId) {
  const latestRun = await prisma.analysisRun.findFirst({
    where: { mediaId },
    orderBy: { runNumber: 'desc' },
    select: { runNumber: true },
  });
  return latestRun?.runNumber || 0;
},
```

---

#### Analysis Result Operations

```javascript
// Save successful analysis result
async createAnalysisResult(runId, resultData) {
  const { modelName, prediction, confidence, resultPayload } = resultData;
  
  // ✨ Extract promoted fields from resultPayload
  const processingTime = resultPayload?.processing_time || null;
  const mediaType = resultPayload?.media_type || null;
  
  return prisma.deepfakeAnalysis.create({
    data: {
      analysisRunId: runId,
      modelName,
      prediction,
      confidence,
      status: 'COMPLETED',
      processingTime,  // Promoted field
      mediaType,       // Promoted field
      resultPayload,
    },
  });
},

// Save failed analysis error
async createAnalysisError(runId, modelName, error) {
  // Ensure errorMessage is always a string
  let errorMessage = String(error.message || error);
  
  // Check if entry already exists (prevents duplicates on BullMQ retries)
  const existingAnalysis = await prisma.deepfakeAnalysis.findFirst({
    where: { analysisRunId: runId, modelName },
  });
  
  const resultPayload = {
    error: errorMessage,
    stack: error.stack,
    serverResponse: error.serverResponse,
    timestamp: new Date().toISOString(),
    retryCount: existingAnalysis 
      ? ((existingAnalysis.resultPayload?.retryCount || 0) + 1) 
      : 0,
  };
  
  if (existingAnalysis) {
    // Update existing entry with new error details
    return prisma.deepfakeAnalysis.update({
      where: { id: existingAnalysis.id },
      data: {
        status: 'FAILED',
        errorMessage,
        resultPayload,
      },
    });
  } else {
    // Create new failed entry
    return prisma.deepfakeAnalysis.create({
      data: {
        analysisRunId: runId,
        modelName,
        status: 'FAILED',
        errorMessage,
        prediction: 'N/A',
        confidence: 0,
        resultPayload,
      },
    });
  }
},
```

---

### User Repository

**File**: `Backend/src/repositories/user.repository.js`

```javascript
const defaultUserSelect = {
  id: true,
  email: true,
  firstName: true,
  lastName: true,
  role: true,
  isActive: true,
  createdAt: true,
  updatedAt: true,
  // password: NEVER exposed
};

export const userRepository = {
  // Find by email WITH password (for login)
  async findByEmailWithPassword(email) {
    return prisma.user.findUnique({ where: { email } });
  },

  // Find by ID WITHOUT password (for profile)
  async findById(userId) {
    return prisma.user.findUnique({
      where: { id: userId },
      select: defaultUserSelect,
    });
  },
  
  // Find by ID WITH password (for password change)
  async findByIdWithPassword(userId) {
    return prisma.user.findUnique({ where: { id: userId } });
  },

  // Create user
  async create(userData) {
    return prisma.user.create({
      data: userData,
      select: defaultUserSelect,
    });
  },

  // Update user
  async update(userId, updateData) {
    return prisma.user.update({
      where: { id: userId },
      data: updateData,
      select: defaultUserSelect,
    });
  },
};
```

**Security Note**: Password is **never** included in default select. Only fetched when explicitly needed (login, password change).

---

## Migrations Strategy

### Prisma Migrate Workflow

```bash
# 1. Modify schema.prisma
# Example: Add new field to Media model

# 2. Generate migration
npx prisma migrate dev --name add_media_metadata

# 3. Prisma generates:
#    - Migration SQL file in prisma/migrations/
#    - Updates Prisma Client

# 4. Apply to production
npx prisma migrate deploy
```

---

### Migration History

#### Major Migrations

**1. Initial Schema** (v1.0)

- User, Video (old name), DeepfakeAnalysis models
- Single-run analysis (no versioning)

**2. Comprehensive Refactor** (`20251009193611`)

- Renamed `Video` → `Media`
- Added `AnalysisRun` for versioning
- Decoupled model enum to string
- Added promoted fields (`processingTime`, `mediaType`)
- Switched to flexible JSON `resultPayload`

**3. Media Metadata** (`20251012034149`)

- Added `metadata` JSON field (FFprobe data)
- Added `hasAudio` boolean field for videos

---

### Migration Example

**File**: `prisma/migrations/20251012034149_add_media_metadata_and_audio/migration.sql`

```sql
-- AlterTable
ALTER TABLE "public"."media" 
ADD COLUMN "has_audio" BOOLEAN,
ADD COLUMN "metadata" JSONB;
```

**Prisma Schema Change**:

```diff
model Media {
  id          String      @id @default(uuid())
  filename    String
  url         String
  // ... other fields
  
+ hasAudio    Boolean?    @map("has_audio")
+ metadata    Json?
}
```

---

### Migration Best Practices

✅ **Always backup** production database before migrating  
✅ **Test migrations** in development/staging first  
✅ **Use descriptive names**: `add_promoted_fields`, not `migration_123`  
✅ **Avoid data loss**: Use `ALTER TABLE ADD COLUMN` (nullable) instead of dropping columns  
✅ **Check constraints**: Ensure unique constraints don't conflict with existing data  
✅ **Rollback plan**: Keep previous migration for quick rollback if needed

---

## Query Patterns

### Common Queries

#### 1. Get User's Media (Newest First)

```javascript
const media = await prisma.media.findMany({
  where: { userId: 'user-uuid' },
  include: {
    analysisRuns: {
      orderBy: { runNumber: 'desc' },
      take: 1,  // Only latest run
      include: { analyses: true },
    },
  },
  orderBy: { createdAt: 'desc' },
});
```

**Performance**: Uses index `[userId, createdAt DESC]`

---

#### 2. Get Media by ID with All Runs

```javascript
const media = await prisma.media.findUnique({
  where: { id: 'media-uuid' },
  include: {
    user: {
      select: { id: true, email: true },
    },
    analysisRuns: {
      orderBy: { runNumber: 'desc' },
      include: {
        analyses: {
          orderBy: { createdAt: 'asc' },
        },
      },
    },
  },
});
```

---

#### 3. Get Latest Run Number

```javascript
const latestRun = await prisma.analysisRun.findFirst({
  where: { mediaId: 'media-uuid' },
  orderBy: { runNumber: 'desc' },
  select: { runNumber: true },
});

const nextRunNumber = (latestRun?.runNumber || 0) + 1;
```

---

#### 4. Get Analyses by Processing Time

```javascript
const fastAnalyses = await prisma.deepfakeAnalysis.findMany({
  where: {
    processingTime: {
      gte: 0,    // Min
      lte: 10,   // Max (10 seconds)
    },
    status: 'COMPLETED',
  },
  orderBy: { processingTime: 'asc' },
  take: 100,
});
```

**Performance**: Uses index `[processingTime]`

---

#### 5. Get Analyses by Media Type

```javascript
const videoAnalyses = await prisma.deepfakeAnalysis.findMany({
  where: { 
    mediaType: 'video',
    status: 'COMPLETED',
  },
  include: {
    analysisRun: {
      include: { media: true },
    },
  },
  orderBy: { createdAt: 'desc' },
  take: 50,
});
```

**Performance**: Uses index `[mediaType]`

---

## Analytics Queries

### Raw SQL Queries

For complex aggregations, Prisma supports raw SQL:

#### 1. Average Processing Time by Model

```javascript
async getAverageProcessingTimeByModel() {
  return prisma.$queryRaw`
    SELECT 
      model_name as "modelName",
      COUNT(*) as "totalAnalyses",
      AVG(processing_time) as "avgProcessingTime",
      MIN(processing_time) as "minProcessingTime",
      MAX(processing_time) as "maxProcessingTime"
    FROM deepfake_analyses
    WHERE processing_time IS NOT NULL 
    AND status = 'COMPLETED'
    GROUP BY model_name
    ORDER BY "avgProcessingTime" ASC
  `;
}
```

**Result**:

```json
[
  {
    "modelName": "SIGLIP-LSTM-V4",
    "totalAnalyses": 150,
    "avgProcessingTime": 42.5,
    "minProcessingTime": 28.1,
    "maxProcessingTime": 78.3
  },
  {
    "modelName": "EFFICIENTNET-B7-V1",
    "totalAnalyses": 120,
    "avgProcessingTime": 55.2,
    "minProcessingTime": 35.4,
    "maxProcessingTime": 95.1
  }
]
```

---

#### 2. Average Confidence by Model and Media Type

```javascript
async getAverageConfidenceByModelAndMediaType() {
  return prisma.$queryRaw`
    SELECT 
      model_name as "modelName",
      media_type as "mediaType",
      COUNT(*) as "totalAnalyses",
      AVG(confidence) as "avgConfidence",
      MIN(confidence) as "minConfidence",
      MAX(confidence) as "maxConfidence"
    FROM deepfake_analyses
    WHERE media_type IS NOT NULL 
    AND status = 'COMPLETED'
    GROUP BY model_name, media_type
    ORDER BY "avgConfidence" DESC
  `;
}
```

**Use Case**: "Which model performs best on videos vs images?"

---

#### 3. Slowest Analyses (Debugging)

```javascript
async getSlowestAnalyses(limit = 10) {
  return prisma.deepfakeAnalysis.findMany({
    where: {
      processingTime: { not: null },
      status: 'COMPLETED',
    },
    include: {
      analysisRun: {
        include: { 
          media: {
            select: { id: true, filename: true, mediaType: true }
          }
        },
      },
    },
    orderBy: { processingTime: 'desc' },
    take: limit,
  });
}
```

**Use Case**: Identify performance bottlenecks

---

## Transaction Handling

### Atomic Operations

Use Prisma transactions for multi-step operations that must succeed/fail together.

#### Example: Create Media and Analysis Run

```javascript
const result = await prisma.$transaction(async (tx) => {
  // 1. Create media record
  const media = await tx.media.create({
    data: {
      filename: 'video.mp4',
      url: 'http://...',
      userId: 'user-uuid',
      // ... other fields
    },
  });

  // 2. Create first analysis run
  const run = await tx.analysisRun.create({
    data: {
      mediaId: media.id,
      runNumber: 1,
      status: 'QUEUED',
    },
  });

  // 3. Create pending analyses for each model
  const analyses = await Promise.all(
    activeModels.map(modelName =>
      tx.deepfakeAnalysis.create({
        data: {
          analysisRunId: run.id,
          modelName,
          status: 'PENDING',
          prediction: '',
          confidence: 0,
          resultPayload: {},
        },
      })
    )
  );

  return { media, run, analyses };
});
```

**Benefits**:

- **Atomicity**: All operations succeed or all fail (rollback)
- **Consistency**: No partial state (e.g., Media without AnalysisRun)
- **Isolation**: Other queries don't see intermediate state

---

### Interactive Transactions

For complex workflows with conditional logic:

```javascript
await prisma.$transaction(async (tx) => {
  const media = await tx.media.findUnique({ where: { id: mediaId } });
  
  if (media.status === 'PROCESSING') {
    throw new Error('Already processing');
  }
  
  await tx.media.update({
    where: { id: mediaId },
    data: { status: 'PROCESSING' },
  });
  
  const run = await tx.analysisRun.create({
    data: { mediaId, runNumber: 2 },
  });
  
  // ... more operations
});
```

---

## Best Practices

### 1. Always Use Repositories

❌ **Bad** (Direct Prisma in controllers):

```javascript
// auth.controller.js
const user = await prisma.user.findUnique({ where: { email } });
```

✅ **Good** (Repository abstraction):

```javascript
// auth.controller.js
const user = await userRepository.findByEmail(email);
```

**Benefits**: Testability, maintainability, DRY

---

### 2. Never Expose Passwords

```javascript
// ✅ Good: Default select excludes password
const defaultUserSelect = {
  id: true,
  email: true,
  // password: NOT included
};

// ❌ Bad: Including entire user object
return prisma.user.findUnique({ where: { id } });
```

---

### 3. Use Promoted Fields for Performance

❌ **Bad** (JSON queries are slow):

```sql
SELECT * FROM deepfake_analyses
WHERE result_payload->>'media_type' = 'video';
```

✅ **Good** (Indexed column query):

```sql
SELECT * FROM deepfake_analyses
WHERE media_type = 'video';
```

---

### 4. Cascade Deletes

✅ Use `onDelete: Cascade` for dependent data:

```prisma
model Media {
  userId String
  user   User @relation(fields: [userId], references: [id], onDelete: Cascade)
}
```

**Automatic cleanup**: Deleting a user removes all their media, runs, and analyses.

---

### 5. Indexes for Common Queries

Add indexes for frequently used `WHERE` and `ORDER BY` columns:

```prisma
@@index([userId, createdAt(sort: Desc)])  // List user's media
@@index([modelName])                       // Filter by model
@@index([processingTime])                  // Sort by performance
```

---

### 6. Pagination for Large Datasets

```javascript
const page = 1;
const limit = 20;
const skip = (page - 1) * limit;

const media = await prisma.media.findMany({
  where: { userId },
  skip,
  take: limit,
  orderBy: { createdAt: 'desc' },
});
```

---

## Summary

The Drishtiksha database schema is:

✅ **Type-Safe** - Prisma generates TypeScript-safe clients  
✅ **Versioned** - Multiple analysis runs per media item  
✅ **Flexible** - JSON columns for dynamic ML responses  
✅ **Performant** - Strategic indexes on common queries  
✅ **Maintainable** - Repository pattern abstracts database access  
✅ **Scalable** - Promoted fields enable efficient analytics  
✅ **Atomic** - Transaction support for complex workflows  
✅ **Clean** - Cascade deletes maintain referential integrity

**Key Models**: User → Media → AnalysisRun → DeepfakeAnalysis  
**Key Pattern**: Repository abstraction with Prisma  
**Key Feature**: Promoted fields + JSON payloads for flexibility + performance

**Next Steps**:

- [Services & Business Logic Documentation](./Services.md)
- [Middleware & Authentication Documentation](./Middleware.md)
- [WebSocket & Real-time Updates Documentation](./WebSocket.md)
