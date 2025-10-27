# Services & Business Logic Layer

Comprehensive documentation for all service modules in the **Drishtiksha Backend**. Services contain the core business logic and orchestrate operations across repositories, storage, and external services.

---

## Table of Contents

1. [Service Architecture Overview](#service-architecture-overview)
2. [Design Principles](#design-principles)
3. [Authentication Service](#authentication-service)
4. [Media Service](#media-service)
5. [Analysis Service](#analysis-service)
6. [Storage Service](#storage-service)
7. [Queue Service](#queue-service)
8. [Server Health Service](#server-health-service)
9. [PDF Report Service](#pdf-report-service)
10. [Event Publisher Service](#event-publisher-service)

---

## Service Architecture Overview

### The Service Layer's Role

The service layer sits between **controllers** (HTTP layer) and **repositories** (data layer). It contains all business logic and workflow orchestration.

```text
┌──────────────┐
│  Controller  │  ← HTTP Request/Response handling
└──────┬───────┘
       │ Calls service methods
       ▼
┌──────────────┐
│   Service    │  ← Business logic & orchestration
└──────┬───────┘
       │ Uses repositories
       ▼
┌──────────────┐
│  Repository  │  ← Database operations
└──────────────┘
```

### Key Characteristics

- **HTTP-Agnostic**: Services never touch `req` or `res` objects
- **Reusable**: Can be called from controllers, workers, or other services
- **Testable**: Easy to unit test in isolation
- **Transaction-Aware**: Manages database transactions when needed

---

## Design Principles

### 1. Single Responsibility

Each service has a clear, focused purpose:

- `authService` handles authentication and authorization
- `mediaService` manages media lifecycle
- `queueService` manages job queueing

### 2. Dependency Injection

Services receive dependencies through constructor or function parameters:

```javascript
// Example: MediaService
class MediaService {
  constructor(mediaRepository, storageManager, queueService) {
    this.mediaRepository = mediaRepository;
    this.storageManager = storageManager;
    this.queueService = queueService;
  }
}
```

### 3. Error Handling

Services throw `ApiError` instances with appropriate status codes:

```javascript
if (!media) {
  throw new ApiError(404, 'Media not found');
}

if (!hasPermission(user, media)) {
  throw new ApiError(403, 'You do not have permission to access this media');
}
```

### 4. Async/Await Pattern

All service methods are asynchronous:

```javascript
async uploadMedia(userId, file, description) {
  const uploadedFile = await this.storageManager.uploadFile(file);
  const media = await this.mediaRepository.create({
    userId,
    filename: uploadedFile.filename,
    url: uploadedFile.url,
    description,
  });
  await this.queueService.addAnalysisJob(media.id);
  return media;
}
```

---

## Authentication Service

**File**: `src/services/auth.service.js`

Handles user authentication, registration, and profile management.

### Methods

#### `signup(userData)`

Creates a new user account with hashed password.

**Parameters**:

- `userData` (object):
  - `email` (string): User's email address
  - `password` (string): Plain text password
  - `firstName` (string): User's first name
  - `lastName` (string): User's last name

**Process**:

1. Check if email already exists
2. Hash password using bcrypt
3. Create user in database
4. Generate JWT token
5. Return user object and token

**Returns**: `{ user, token }`

**Throws**:

- `ApiError(409)`: Email already in use

**Implementation**:

```javascript
async signup(email, password, firstName, lastName) {
  // Check existing user
  const existingUser = await userRepository.findByEmail(email);
  if (existingUser) {
    throw new ApiError(409, 'Email is already in use');
  }

  // Hash password
  const hashedPassword = await bcrypt.hash(password, 10);

  // Create user
  const user = await userRepository.create({
    email,
    password: hashedPassword,
    firstName,
    lastName,
  });

  // Generate token
  const token = jwt.sign(
    { userId: user.id },
    process.env.JWT_SECRET,
    { expiresIn: process.env.JWT_EXPIRES_IN || '7d' }
  );

  return { user, token };
}
```

---

#### `login(email, password)`

Authenticates user and returns JWT token.

**Parameters**:

- `email` (string): User's email
- `password` (string): Plain text password

**Process**:

1. Find user by email
2. Verify password using bcrypt
3. Generate JWT token
4. Return user and token

**Returns**: `{ user, token }`

**Throws**:

- `ApiError(401)`: Invalid credentials

**Implementation**:

```javascript
async login(email, password) {
  // Find user
  const user = await userRepository.findByEmail(email);
  if (!user) {
    throw new ApiError(401, 'Invalid email or password');
  }

  // Verify password
  const isPasswordValid = await bcrypt.compare(password, user.password);
  if (!isPasswordValid) {
    throw new ApiError(401, 'Invalid email or password');
  }

  // Generate token
  const token = jwt.sign(
    { userId: user.id },
    process.env.JWT_SECRET,
    { expiresIn: process.env.JWT_EXPIRES_IN || '7d' }
  );

  return { user, token };
}
```

---

#### `getUserProfile(userId)`

Retrieves user profile by ID.

**Parameters**:

- `userId` (string): UUID of user

**Returns**: User object

**Throws**:

- `ApiError(404)`: User not found

---

#### `updateProfile(userId, updates)`

Updates user's first name and/or last name.

**Parameters**:

- `userId` (string): UUID of user
- `updates` (object): Fields to update
  - `firstName` (string, optional)
  - `lastName` (string, optional)

**Returns**: Updated user object

---

#### `updatePassword(userId, currentPassword, newPassword)`

Changes user's password after verification.

**Parameters**:

- `userId` (string): UUID of user
- `currentPassword` (string): Current password for verification
- `newPassword` (string): New password to set

**Process**:

1. Fetch user from database
2. Verify current password
3. Hash new password
4. Update password in database

**Returns**: void

**Throws**:

- `ApiError(401)`: Current password is incorrect

---

## Media Service

**File**: `src/services/media.service.js`

Orchestrates media upload, storage, metadata extraction, and analysis workflow.

### Dependencies

- `mediaRepository`: Database operations
- `analysisRunRepository`: Analysis run management
- `storageManager`: File storage (local/cloud)
- `queueService`: Job queue management
- `eventPublisher`: Real-time event publishing
- `mlServerClient`: ML server communication

### Media Service Methods

#### `uploadMedia(userId, file, description)`

Handles complete media upload and analysis initiation workflow.

**Parameters**:

- `userId` (string): Owner's UUID
- `file` (object): Multer file object
  - `buffer` (Buffer): File data
  - `originalname` (string): Original filename
  - `mimetype` (string): MIME type
  - `size` (number): File size in bytes
- `description` (string, optional): User-provided description

**Process Flow**:

1. **Upload to storage**: Upload file to configured storage provider
2. **Extract metadata**: Use FFprobe for video/audio metadata
3. **Create database record**: Store media info in PostgreSQL
4. **Create initial analysis run**: Create first AnalysisRun record
5. **Queue analysis jobs**: Add BullMQ flow with child jobs per model
6. **Publish event**: Emit `media_uploaded` event via Socket.IO
7. **Return media object**: Immediate response with QUEUED status

**Returns**: Media object with initial AnalysisRun

**Implementation**:

```javascript
async uploadMedia(userId, file, description = null) {
  // 1. Upload to storage
  const uploadResult = await storageManager.uploadFile(file);

  // 2. Extract metadata
  const metadata = await extractMediaMetadata(uploadResult.path);

  // 3. Detect media type
  const mediaType = detectMediaType(file.mimetype);

  // 4. Create database record
  const media = await mediaRepository.create({
    userId,
    filename: uploadResult.filename,
    url: uploadResult.url,
    publicId: uploadResult.publicId,
    mimetype: file.mimetype,
    size: file.size,
    description,
    status: 'QUEUED',
    mediaType,
    hasAudio: metadata.hasAudio,
    metadata,
  });

  // 5. Create initial analysis run
  const analysisRun = await analysisRunRepository.create({
    mediaId: media.id,
    runNumber: 1,
    status: 'QUEUED',
  });

  // 6. Queue analysis jobs
  await queueService.addMediaAnalysisFlow(media.id, analysisRun.id);

  // 7. Publish event
  await eventPublisher.publish('media_uploaded', {
    userId,
    mediaId: media.id,
    filename: media.filename,
  });

  return { ...media, latestAnalysisRunId: analysisRun.id };
}
```

---

#### `getAllUserMedia(userId)`

Retrieves all media items for a user with their analysis runs.

**Parameters**:

- `userId` (string): User's UUID

**Returns**: Array of media objects with nested `analysisRuns`

**Database Query**:

```javascript
const mediaList = await mediaRepository.findAllByUserId(userId, {
  include: {
    analysisRuns: {
      orderBy: { createdAt: 'desc' },
      include: {
        analyses: {
          select: {
            id: true,
            modelName: true,
            prediction: true,
            confidence: true,
            status: true,
          },
        },
      },
    },
  },
  orderBy: { createdAt: 'desc' },
});
```

---

#### `getMediaById(mediaId, userId)`

Retrieves single media item with full analysis details.

**Parameters**:

- `mediaId` (string): Media UUID
- `userId` (string): Requesting user's UUID

**Process**:

1. Fetch media with all relations
2. Verify user owns the media
3. Return complete media object

**Returns**: Media object with nested analysis runs and analyses

**Throws**:

- `ApiError(404)`: Media not found
- `ApiError(403)`: User doesn't own media

---

#### `updateMedia(mediaId, userId, updates)`

Updates media description.

**Parameters**:

- `mediaId` (string): Media UUID
- `userId` (string): User's UUID
- `updates` (object): Fields to update
  - `description` (string, optional)

**Returns**: Updated media object

---

#### `deleteMedia(mediaId, userId)`

Deletes media file and all associated database records.

**Parameters**:

- `mediaId` (string): Media UUID
- `userId` (string): User's UUID

**Process**:

1. Verify ownership
2. Delete file from storage
3. Delete database record (cascades to analysis runs/analyses)
4. Publish event

**Returns**: void

**Side Effects**:

- Deletes physical file
- Cascades to all `AnalysisRun` and `DeepfakeAnalysis` records

---

#### `reanalyzeMedia(mediaId, userId)`

Creates new analysis run for existing media.

**Parameters**:

- `mediaId` (string): Media UUID
- `userId` (string): User's UUID

**Process**:

1. Verify ownership
2. Get next run number
3. Create new AnalysisRun
4. Queue new analysis flow
5. Update media status to QUEUED

**Returns**: Media object with new AnalysisRun

**Use Cases**:

- Test new models on existing media
- Re-analyze after ML server updates
- Recover from failed analysis

---

## Analysis Service

**File**: `src/services/analysis.service.js`

Manages deepfake analysis execution and result storage.

### Analysis Service Methods

#### `runSingleModelAnalysis(mediaId, analysisRunId, modelName)`

Executes analysis for a single model.

**Parameters**:

- `mediaId` (string): Media UUID
- `analysisRunId` (string): AnalysisRun UUID
- `modelName` (string): Model identifier (e.g., 'SIGLIP-LSTM-V4')

**Process**:

1. Fetch media record
2. Download file if cloud storage
3. Create DeepfakeAnalysis record (status: PROCESSING)
4. Send file to ML server
5. Parse and save results
6. Update analysis status to COMPLETED
7. Publish progress event

**Returns**: DeepfakeAnalysis object

**Error Handling**:

```javascript
try {
  const result = await mlServerClient.analyze(filePath, modelName);
  await analysisRepository.updateResultPayload(analysisId, result);
  await analysisRepository.updateStatus(analysisId, 'COMPLETED');
} catch (error) {
  await analysisRepository.updateStatus(analysisId, 'FAILED');
  await analysisRepository.createError(analysisId, {
    errorMessage: error.message,
    errorStack: error.stack,
  });
  throw error;
}
```

---

#### `finalizeAnalysisRun(analysisRunId)`

Updates AnalysisRun and Media status after all model analyses complete.

**Parameters**:

- `analysisRunId` (string): AnalysisRun UUID

**Process**:

1. Fetch all analyses for the run
2. Count completed vs failed
3. Determine final status:
   - `ANALYZED`: All succeeded
   - `PARTIALLY_ANALYZED`: Some succeeded
   - `FAILED`: All failed
4. Update AnalysisRun status
5. Update Media status
6. Publish completion event

**Returns**: Updated AnalysisRun object

**Status Logic**:

```javascript
const totalAnalyses = analyses.length;
const completedCount = analyses.filter(a => a.status === 'COMPLETED').length;
const failedCount = analyses.filter(a => a.status === 'FAILED').length;

let finalStatus;
if (completedCount === totalAnalyses) {
  finalStatus = 'ANALYZED';
} else if (completedCount > 0) {
  finalStatus = 'PARTIALLY_ANALYZED';
} else {
  finalStatus = 'FAILED';
}
```

---

## Storage Service

**File**: `src/storage/storage.manager.js`

Abstracts file storage operations across different providers.

### Architecture

The storage manager uses the **Strategy Pattern** to support multiple storage backends:

```javascript
// storage.manager.js
const STORAGE_PROVIDER = process.env.STORAGE_PROVIDER || 'local';

let storageProvider;
if (STORAGE_PROVIDER === 'cloudinary') {
  storageProvider = await import('./cloudinary.provider.js');
} else {
  storageProvider = await import('./local.provider.js');
}

export const { uploadFile, deleteFile, getFileUrl } = storageProvider;
```

### Provider Interface

All storage providers must implement:

```typescript
interface StorageProvider {
  uploadFile(file: MulterFile): Promise<UploadResult>;
  deleteFile(publicId: string): Promise<void>;
  getFileUrl(publicId: string): string;
}

interface UploadResult {
  filename: string;
  url: string;
  publicId: string;
  size: number;
}
```

### Local Storage Provider

**File**: `src/storage/local.provider.js`

Stores files in local filesystem.

**Upload Flow**:

```javascript
export async function uploadFile(file) {
  const mediaType = detectMediaType(file.mimetype);
  const subfolder = `${mediaType.toLowerCase()}s`; // 'videos', 'audios', 'images'
  
  const filename = `${Date.now()}-${file.originalname}`;
  const relativePath = `media/${subfolder}/${filename}`;
  const absolutePath = path.join(process.cwd(), 'public', relativePath);
  
  await fs.mkdir(path.dirname(absolutePath), { recursive: true });
  await fs.writeFile(absolutePath, file.buffer);
  
  return {
    filename,
    url: `${process.env.BACKEND_URL}/${relativePath}`,
    publicId: relativePath,
    size: file.size,
  };
}
```

**Delete Flow**:

```javascript
export async function deleteFile(publicId) {
  const absolutePath = path.join(process.cwd(), 'public', publicId);
  try {
    await fs.unlink(absolutePath);
  } catch (error) {
    if (error.code !== 'ENOENT') throw error; // Ignore if already deleted
  }
}
```

### Cloudinary Provider

**File**: `src/storage/cloudinary.provider.js`

Stores files in Cloudinary cloud storage.

**Configuration**:

```javascript
import { v2 as cloudinary } from 'cloudinary';

cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
});
```

**Upload Flow**:

```javascript
export async function uploadFile(file) {
  const mediaType = detectMediaType(file.mimetype);
  const resourceType = mediaType === 'VIDEO' ? 'video' : 'auto';
  
  const result = await cloudinary.uploader.upload_stream(
    {
      folder: `drishtiksha/${mediaType.toLowerCase()}s`,
      resource_type: resourceType,
      use_filename: true,
    },
    (error, result) => {
      if (error) throw error;
      return result;
    }
  ).end(file.buffer);
  
  return {
    filename: result.original_filename,
    url: result.secure_url,
    publicId: result.public_id,
    size: result.bytes,
  };
}
```

---

## Queue Service

**File**: `src/services/queue.service.js`

Manages BullMQ job queue operations.

### Queue Service Architecture

Uses BullMQ Flows to create parent-child job hierarchies:

```text
Parent Job: finalize-analysis-run-{runId}
├── Child 1: run-single-analysis (Model: SIGLIP-LSTM-V4)
├── Child 2: run-single-analysis (Model: COLOR-CUES-LSTM-V1)
└── Child 3: run-single-analysis (Model: EFFICIENTNET-B7-V1)

Parent only runs after ALL children complete/fail
```

### Queue Service Methods

#### `addMediaAnalysisFlow(mediaId, analysisRunId)`

Creates BullMQ flow for multi-model analysis.

**Parameters**:

- `mediaId` (string): Media UUID
- `analysisRunId` (string): AnalysisRun UUID

**Process**:

1. Query ML server for compatible models
2. Create child job for each model
3. Create parent finalizer job
4. Add flow to queue

**Implementation**:

```javascript
async addMediaAnalysisFlow(mediaId, analysisRunId) {
  // Get compatible models from ML server
  const media = await mediaRepository.findById(mediaId);
  const compatibleModels = await mlServerClient.getCompatibleModels(media.mediaType);
  
  // Create child jobs
  const childrenJobs = compatibleModels.map(modelName => ({
    name: 'run-single-analysis',
    data: { mediaId, analysisRunId, modelName },
    opts: {
      attempts: 3,
      backoff: { type: 'exponential', delay: 5000 },
    },
  }));
  
  // Create parent job
  const parentJob = {
    name: 'finalize-analysis-run',
    data: { mediaId, analysisRunId },
    queueName: 'media-processing',
    children: childrenJobs,
  };
  
  // Add flow to queue
  await flowProducer.add(parentJob);
}
```

---

#### `getQueueStats()`

Retrieves current queue statistics.

**Returns**:

```javascript
{
  pending: number,   // Waiting to be processed
  active: number,    // Currently processing
  completed: number, // Successfully finished
  failed: number,    // Errored during processing
  delayed: number,   // Scheduled for future
}
```

**Implementation**:

```javascript
async getQueueStats() {
  const queue = getQueue('media-processing');
  
  const [pending, active, completed, failed, delayed] = await Promise.all([
    queue.getWaitingCount(),
    queue.getActiveCount(),
    queue.getCompletedCount(),
    queue.getFailedCount(),
    queue.getDelayedCount(),
  ]);
  
  return { pending, active, completed, failed, delayed };
}
```

---

## Server Health Service

**File**: `src/services/server-health.service.js`

Monitors ML server health and stores historical records.

### Server Health Methods

#### `checkServerHealth()`

Pings ML server and records response.

**Process**:

1. Send GET request to `/stats` endpoint
2. Measure response time
3. Parse response payload
4. Determine health status
5. Save to database
6. Cache result in Redis (60s TTL)

**Returns**: ServerHealth object

**Implementation**:

```javascript
async checkServerHealth() {
  const startTime = Date.now();
  
  try {
    const response = await axios.get(
      `${process.env.SERVER_URL}/stats`,
      { timeout: 10000 }
    );
    
    const responseTimeMs = Date.now() - startTime;
    
    const healthRecord = await serverHealthRepository.create({
      status: 'HEALTHY',
      responseTimeMs,
      statsPayload: response.data,
      checkedAt: new Date(),
    });
    
    // Cache for 60 seconds
    await redis.setex(
      'server_health_cache',
      60,
      JSON.stringify(healthRecord)
    );
    
    return healthRecord;
  } catch (error) {
    const responseTimeMs = Date.now() - startTime;
    
    return await serverHealthRepository.create({
      status: 'UNHEALTHY',
      responseTimeMs,
      statsPayload: { error: error.message },
      checkedAt: new Date(),
    });
  }
}
```

---

#### `getHealthHistory(limit = 50)`

Retrieves recent health check records.

**Parameters**:

- `limit` (number): Max records to return

**Returns**: Array of ServerHealth objects

---

## PDF Report Service

**File**: `src/services/pdf.service.js`

Generates PDF reports from analysis results.

### PDF Service Methods

#### `generateAnalysisRunReport(analysisRunId, userId)`

Creates PDF report for an analysis run.

**Parameters**:

- `analysisRunId` (string): AnalysisRun UUID
- `userId` (string): Requesting user's UUID

**Process**:

1. Verify user permission
2. Fetch analysis run with all relations
3. Generate markdown from template
4. Convert markdown to PDF
5. Return PDF buffer

**Returns**: PDF Buffer

**Template Structure**:

```markdown
# Deepfake Analysis Report

## Executive Summary
- Media: {filename}
- Analysis Run: #{runNumber}
- Overall Status: {status}
- Total Models: {totalModels}
- Completed: {completedCount}
- Failed: {failedCount}

## Model Results

| Model | Prediction | Confidence | Status |
|-------|-----------|------------|--------|
| {modelName} | {prediction} | {confidence} | {status} |

## Detailed Analysis

### {modelName}
- **Prediction**: {prediction}
- **Confidence**: {confidence}
- **Processing Time**: {processingTime}s
- **Result Payload**: ```json
{resultPayload}
```

## Metadata

- Upload Date: {createdAt}
- Analysis Date: {analyzedAt}
- User: {userEmail}

---

## Event Publisher Service

**File**: `src/services/event-publisher.service.js`

Publishes real-time events via Redis Pub/Sub.

### Event Publisher Architecture

```text
┌─────────────┐         Publish          ┌─────────┐
│   Worker    │─────────────────────────►│  Redis  │
└─────────────┘                           └────┬────┘
                                               │ Subscribe
┌─────────────┐                           ┌────▼────┐
│ API Server  │◄──────────────────────────│  Redis  │
└──────┬──────┘                           └─────────┘
       │ Emit via Socket.IO
       ▼
┌─────────────┐
│   Client    │
└─────────────┘
```

### Event Publisher Methods

#### `publish(event, data)`

Publishes event to Redis channel.

**Parameters**:

- `event` (string): Event name
- `data` (object): Event payload

**Implementation**:

```javascript
async publish(event, data) {
  const payload = JSON.stringify({
    event,
    data,
    timestamp: new Date().toISOString(),
  });
  
  await redis.publish('media-progress-events', payload);
}
```

### Event Types

| Event | Published By | Data |
|-------|-------------|------|
| `media_uploaded` | API Server | `{ userId, mediaId, filename }` |
| `analysis_started` | Worker | `{ mediaId, runId, modelName }` |
| `analysis_progress` | Worker | `{ mediaId, runId, progress }` |
| `analysis_completed` | Worker | `{ mediaId, runId, modelName, result }` |
| `analysis_failed` | Worker | `{ mediaId, runId, modelName, error }` |
| `run_finalized` | Worker | `{ mediaId, runId, finalStatus }` |

---

## Service Interaction Examples

### Complete Upload-to-Analysis Flow

```javascript
// Controller layer
async function uploadMediaController(req, res) {
  const { userId } = req.user;
  const { file } = req;
  const { description } = req.body;
  
  // Delegate to service
  const media = await mediaService.uploadMedia(userId, file, description);
  
  res.status(202).json(new ApiResponse(
    202,
    media,
    'Media uploaded and successfully queued'
  ));
}

// Service layer (mediaService.uploadMedia)
async uploadMedia(userId, file, description) {
  // 1. Upload file
  const uploadResult = await storageManager.uploadFile(file);
  
  // 2. Extract metadata
  const metadata = await extractMetadata(uploadResult.path);
  
  // 3. Create DB record
  const media = await mediaRepository.create({ ... });
  
  // 4. Create analysis run
  const run = await analysisRunRepository.create({ ... });
  
  // 5. Queue jobs
  await queueService.addMediaAnalysisFlow(media.id, run.id);
  
  // 6. Publish event
  await eventPublisher.publish('media_uploaded', { ... });
  
  return media;
}

// Worker picks up job
worker.on('run-single-analysis', async (job) => {
  const { mediaId, analysisRunId, modelName } = job.data;
  
  // Execute analysis
  await analysisService.runSingleModelAnalysis(
    mediaId,
    analysisRunId,
    modelName
  );
});

// Worker picks up finalizer
worker.on('finalize-analysis-run', async (job) => {
  const { analysisRunId } = job.data;
  
  // Finalize run
  await analysisService.finalizeAnalysisRun(analysisRunId);
});
```

---

## Testing Services

### Unit Testing Example

```javascript
// tests/services/auth.service.test.js
describe('AuthService', () => {
  describe('signup', () => {
    it('should create user and return token', async () => {
      const userData = {
        email: 'test@example.com',
        password: 'password123',
        firstName: 'John',
        lastName: 'Doe',
      };
      
      const result = await authService.signup(userData);
      
      expect(result).toHaveProperty('user');
      expect(result).toHaveProperty('token');
      expect(result.user.email).toBe(userData.email);
    });
    
    it('should throw error for duplicate email', async () => {
      await expect(
        authService.signup({ email: 'existing@example.com', ... })
      ).rejects.toThrow('Email is already in use');
    });
  });
});
```

---

## Best Practices

### 1. Keep Services Focused

❌ **Bad** - Service doing too much:

```javascript
async uploadAndAnalyzeAndNotify(file) {
  const upload = await upload(file);
  const analysis = await analyze(upload);
  await sendEmail(analysis);
  await sendSMS(analysis);
  return analysis;
}
```

✅ **Good** - Separate concerns:

```javascript
async uploadMedia(file) {
  const upload = await storageManager.uploadFile(file);
  const media = await mediaRepository.create(upload);
  await queueService.addAnalysisJob(media.id);
  return media;
}
```

### 2. Handle Errors Gracefully

```javascript
async deleteMedia(mediaId, userId) {
  const media = await mediaRepository.findById(mediaId);
  
  if (!media) {
    throw new ApiError(404, 'Media not found');
  }
  
  if (media.userId !== userId) {
    throw new ApiError(403, 'Unauthorized');
  }
  
  try {
    await storageManager.deleteFile(media.publicId);
  } catch (error) {
    // Log but don't fail if file already deleted
    logger.warn(`File ${media.publicId} not found in storage`);
  }
  
  await mediaRepository.delete(mediaId);
}
```

### 3. Use Transactions for Multi-Step Operations

```javascript
async createMediaWithAnalysisRun(data) {
  return await prisma.$transaction(async (tx) => {
    const media = await tx.media.create({ data });
    const run = await tx.analysisRun.create({
      data: { mediaId: media.id, runNumber: 1 },
    });
    return { media, run };
  });
}
```

---

## Summary

The service layer provides:

✅ **Clear Separation** - Business logic isolated from HTTP and data layers  
✅ **Reusability** - Services called from controllers, workers, other services  
✅ **Testability** - Easy to unit test in isolation  
✅ **Maintainability** - Changes localized to specific services  
✅ **Flexibility** - Easy to swap implementations (storage providers)  
✅ **Error Handling** - Consistent error throwing with ApiError  
✅ **Async Operations** - All methods return Promises  
✅ **Event-Driven** - Real-time updates via event publisher

**Next Steps**:

- [Middleware & Authentication](./Middleware.md)
- [WebSocket & Real-time Updates](./WebSocket.md)
- [Database Schema](./Database-Schema.md)
