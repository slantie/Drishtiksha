# API Routes & Endpoints

Complete reference for all REST API endpoints in the **Drishtiksha Backend**. All endpoints use JSON for request/response bodies and follow RESTful conventions.

---

## Table of Contents

1. [Base URL & Versioning](#base-url--versioning)
2. [Authentication Flow](#authentication-flow)
3. [Common Response Format](#common-response-format)
4. [Error Handling](#error-handling)
5. [Authentication Endpoints](#authentication-endpoints)
6. [Media Management Endpoints](#media-management-endpoints)
7. [Monitoring & Health Endpoints](#monitoring--health-endpoints)
8. [PDF Report Endpoints](#pdf-report-endpoints)
9. [Rate Limiting](#rate-limiting)
10. [Request Validation](#request-validation)

---

## Base URL & Versioning

**Base URL**: `http://localhost:3000/api/v1`  
**Version**: v1 (current)  
**Content-Type**: `application/json` (except file uploads: `multipart/form-data`)

All endpoints are prefixed with `/api/v1` for versioning. Future breaking changes will increment the version (v2, v3, etc.).

---

## Authentication Flow

### JWT-Based Authentication

```text
┌──────────────┐                                    ┌──────────────┐
│   Client     │                                    │   Backend    │
└──────┬───────┘                                    └──────┬───────┘
       │                                                   │
       │  1. POST /api/v1/auth/signup                     │
       │     { email, password, firstName, lastName }     │
       ├──────────────────────────────────────────────────►
       │                                                   │
       │  2. Response: { token, user }                    │
       │◄──────────────────────────────────────────────────┤
       │                                                   │
       │  3. Store token in localStorage                  │
       │     or sessionStorage                             │
       │                                                   │
       │  4. All subsequent requests include:             │
       │     Authorization: Bearer <token>                │
       ├──────────────────────────────────────────────────►
       │                                                   │
       │  5. Backend verifies token via                   │
       │     authenticateToken middleware                 │
       │                                                   │
       │  6. Attaches user to req.user                    │
       │◄──────────────────────────────────────────────────┤
```

**Token Storage**: Store JWT token in `localStorage` or `sessionStorage`  
**Token Header**: `Authorization: Bearer <token>`  
**Token Expiration**: Configurable via `JWT_EXPIRES_IN` (default: 7 days)

---

## Common Response Format

### Success Response

All successful API responses follow this structure:

```json
{
  "statusCode": 200,
  "success": true,
  "message": "Operation completed successfully",
  "data": {
    // Response payload (varies by endpoint)
  }
}
```

**ApiResponse Class** (`src/utils/ApiResponse.js`):

```javascript
class ApiResponse {
  constructor(statusCode, data, message = "Success") {
    this.statusCode = statusCode;
    this.data = data;
    this.message = message;
    this.success = statusCode < 400;
  }
}
```

### Common HTTP Status Codes

| Code | Meaning | When Used |
|------|---------|-----------|
| `200` | OK | Successful GET/PUT/PATCH/DELETE |
| `201` | Created | Successful POST (resource created) |
| `202` | Accepted | Request accepted for async processing |
| `400` | Bad Request | Validation errors, invalid input |
| `401` | Unauthorized | Missing/invalid JWT token |
| `403` | Forbidden | User lacks permission |
| `404` | Not Found | Resource doesn't exist |
| `415` | Unsupported Media Type | Invalid file type |
| `429` | Too Many Requests | Rate limit exceeded |
| `500` | Internal Server Error | Server-side error |
| `503` | Service Unavailable | External service (ML Server) down |

---

## Error Handling

### Error Response Format

```json
{
  "statusCode": 400,
  "success": false,
  "message": "Validation failed",
  "errors": [
    "Email is required",
    "Password must be at least 6 characters"
  ]
}
```

### ApiError Class

```javascript
// src/utils/ApiError.js
class ApiError extends Error {
  constructor(statusCode, message, errors = null) {
    super(message);
    this.statusCode = statusCode;
    this.errors = errors;
    this.isOperational = true;
    Error.captureStackTrace(this, this.constructor);
  }
}
```

### Error Middleware

```javascript
// src/middleware/error.middleware.js
export const errorMiddleware = (err, req, res, next) => {
  let { statusCode, message, errors } = err;

  if (!statusCode) statusCode = 500;
  if (!message) message = "Internal Server Error";

  // Log 5xx errors
  if (statusCode >= 500) {
    logger.error(`[${req.method}] ${req.path} >> ${statusCode}: ${message}`);
    logger.error(err.stack);
  }

  res.status(statusCode).json({
    success: false,
    message,
    errors,
    ...(process.env.NODE_ENV === "development" && { stack: err.stack })
  });
};
```

---

## Authentication Endpoints

**Base Path**: `/api/v1/auth`

### POST /auth/signup

Create a new user account.

**Request**:

```http
POST /api/v1/auth/signup
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securePassword123",
  "firstName": "John",
  "lastName": "Doe"
}
```

**Validation** (Zod Schema):

```javascript
signupSchema = z.object({
  body: z.object({
    email: z.string().email('Invalid email address'),
    password: z.string().min(6, 'Password must be at least 6 characters long'),
    firstName: z.string().min(1, 'First name is required'),
    lastName: z.string().min(1, 'Last name is required'),
  }),
});
```

**Response** (201 Created):

```json
{
  "statusCode": 201,
  "success": true,
  "message": "User created successfully",
  "data": {
    "user": {
      "id": "uuid-here",
      "email": "user@example.com",
      "firstName": "John",
      "lastName": "Doe",
      "role": "USER",
      "isActive": true,
      "createdAt": "2025-10-26T10:30:00.000Z"
    },
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  }
}
```

**Errors**:

- `400`: Validation failed (duplicate email, weak password)
- `500`: Server error (database connection failed)

---

### POST /auth/login

Authenticate user and receive JWT token.

**Request**:

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securePassword123"
}
```

**Rate Limiting**: Max 10 failed attempts per 10 minutes (see [Rate Limiting](#rate-limiting))

**Validation**:

```javascript
loginSchema = z.object({
  body: z.object({
    email: z.string().email('A valid email is required'),
    password: z.string().min(1, 'Password is required'),
  }),
});
```

**Response** (200 OK):

```json
{
  "statusCode": 200,
  "success": true,
  "message": "Login successful",
  "data": {
    "user": {
      "id": "uuid-here",
      "email": "user@example.com",
      "firstName": "John",
      "lastName": "Doe",
      "role": "USER"
    },
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  }
}
```

**Errors**:

- `400`: Validation failed (invalid email format)
- `401`: Invalid email or password
- `429`: Too many failed login attempts (rate limit)

---

### POST /auth/logout

Logout user (client-side token invalidation).

**Request**:

```http
POST /api/v1/auth/logout
Authorization: Bearer <token>
```

**Response** (200 OK):

```json
{
  "statusCode": 200,
  "success": true,
  "message": "Logout successful",
  "data": null
}
```

**Note**: This is a client-side operation. The server doesn't maintain a token blacklist. Clients should delete the token from storage.

---

### GET /auth/profile

Get current user's profile.

**Authentication**: Required (`authenticateToken` middleware)

**Request**:

```http
GET /api/v1/auth/profile
Authorization: Bearer <token>
```

**Response** (200 OK):

```json
{
  "statusCode": 200,
  "success": true,
  "message": "Profile retrieved successfully",
  "data": {
    "id": "uuid-here",
    "email": "user@example.com",
    "firstName": "John",
    "lastName": "Doe",
    "role": "USER",
    "isActive": true,
    "createdAt": "2025-10-26T10:30:00.000Z",
    "updatedAt": "2025-10-26T10:30:00.000Z"
  }
}
```

**Errors**:

- `401`: Invalid or expired token
- `404`: User not found

---

### PUT /auth/profile

Update user profile (first name, last name).

**Authentication**: Required

**Request**:

```http
PUT /api/v1/auth/profile
Authorization: Bearer <token>
Content-Type: application/json

{
  "firstName": "Jane",
  "lastName": "Smith"
}
```

**Validation**:

```javascript
updateProfileSchema = z.object({
  body: z.object({
    firstName: z.string().min(1).optional(),
    lastName: z.string().min(1).optional(),
  }),
});
```

**Response** (200 OK):

```json
{
  "statusCode": 200,
  "success": true,
  "message": "Profile updated successfully",
  "data": {
    "id": "uuid-here",
    "email": "user@example.com",
    "firstName": "Jane",
    "lastName": "Smith",
    "role": "USER",
    "updatedAt": "2025-10-26T11:00:00.000Z"
  }
}
```

---

### PUT /auth/profile/password

Change user password.

**Authentication**: Required

**Request**:

```http
PUT /api/v1/auth/profile/password
Authorization: Bearer <token>
Content-Type: application/json

{
  "currentPassword": "oldPassword123",
  "newPassword": "newSecurePassword456"
}
```

**Validation**:

```javascript
updatePasswordSchema = z.object({
  body: z.object({
    currentPassword: z.string().min(1, 'Current password is required'),
    newPassword: z.string().min(6, 'New password must be at least 6 characters'),
  }),
});
```

**Response** (200 OK):

```json
{
  "statusCode": 200,
  "success": true,
  "message": "Password updated successfully",
  "data": null
}
```

**Errors**:

- `401`: Current password is incorrect
- `400`: New password too weak

---

## Media Management Endpoints

**Base Path**: `/api/v1/media`  
**Authentication**: All endpoints require valid JWT token

### POST /media

Upload media file for deepfake analysis.

**Request**:

```http
POST /api/v1/media
Authorization: Bearer <token>
Content-Type: multipart/form-data

{
  "media": <file binary data>,
  "description": "Optional description of the media"
}
```

**Supported File Types**:

- **Video**: `.mp4`, `.avi`, `.mov`, `.mkv`
- **Audio**: `.mp3`, `.wav`, `.ogg`, `.m4a`
- **Image**: `.jpg`, `.jpeg`, `.png`, `.webp`

**File Size Limit**: Configured in multer middleware (typically 100MB)

**Response** (202 Accepted):

```json
{
  "statusCode": 202,
  "success": true,
  "message": "Media uploaded and successfully queued for its first analysis run.",
  "data": {
    "id": "media-uuid",
    "filename": "video.mp4",
    "url": "http://localhost:3001/media/videos/video-uuid.mp4",
    "publicId": "videos/video-uuid",
    "mimetype": "video/mp4",
    "size": 15728640,
    "description": "Optional description",
    "status": "QUEUED",
    "mediaType": "VIDEO",
    "hasAudio": true,
    "metadata": {
      "format": {
        "duration": 30.5,
        "bit_rate": 4128000
      },
      "video": {
        "codec": "h264",
        "width": 1920,
        "height": 1080,
        "fps": 30
      },
      "audio": {
        "codec": "aac",
        "sample_rate": 48000
      }
    },
    "userId": "user-uuid",
    "createdAt": "2025-10-26T12:00:00.000Z",
    "latestAnalysisRunId": null
  }
}
```

**Process Flow**:

1. **Upload**: File saved to temp directory via multer
2. **Storage**: File uploaded to local storage or Cloudinary
3. **Metadata Extraction**: FFprobe extracts video/audio metadata
4. **Database Record**: Create `Media` record in PostgreSQL
5. **Queue Job**: Add analysis job to BullMQ queue (status: `QUEUED`)
6. **Background Processing**: Worker processes job asynchronously
7. **WebSocket Updates**: Real-time progress sent to client

**Errors**:

- `400`: No file provided, invalid file type
- `415`: Unsupported media type
- `500`: Upload failed, storage service unavailable

---

### GET /media

Get all media items for authenticated user.

**Request**:

```http
GET /api/v1/media
Authorization: Bearer <token>
```

**Response** (200 OK):

```json
{
  "statusCode": 200,
  "success": true,
  "message": "Success",
  "data": [
    {
      "id": "media-uuid-1",
      "filename": "video1.mp4",
      "url": "http://localhost:3001/media/videos/video1.mp4",
      "status": "ANALYZED",
      "mediaType": "VIDEO",
      "hasAudio": true,
      "createdAt": "2025-10-26T10:00:00.000Z",
      "latestAnalysisRunId": "run-uuid-1",
      "analysisRuns": [
        {
          "id": "run-uuid-1",
          "runNumber": 1,
          "status": "ANALYZED",
          "createdAt": "2025-10-26T10:00:00.000Z",
          "analyses": [
            {
              "id": "analysis-uuid-1",
              "modelName": "SIGLIP-LSTM-V4",
              "prediction": "FAKE",
              "confidence": 0.92,
              "status": "COMPLETED"
            }
          ]
        }
      ]
    }
  ]
}
```

---

### GET /media/:id

Get specific media item with all analysis runs.

**Request**:

```http
GET /api/v1/media/abc123-uuid
Authorization: Bearer <token>
```

**Validation**:

```javascript
mediaIdParamSchema = z.object({
  params: z.object({
    id: z.string().uuid('Invalid media ID format.'),
  }),
});
```

**Response** (200 OK):

```json
{
  "statusCode": 200,
  "success": true,
  "message": "Success",
  "data": {
    "id": "abc123-uuid",
    "filename": "video.mp4",
    "url": "http://localhost:3001/media/videos/video.mp4",
    "publicId": "videos/video-uuid",
    "mimetype": "video/mp4",
    "size": 15728640,
    "description": "Test video",
    "status": "ANALYZED",
    "mediaType": "VIDEO",
    "hasAudio": true,
    "metadata": { /* FFprobe metadata */ },
    "userId": "user-uuid",
    "createdAt": "2025-10-26T10:00:00.000Z",
    "updatedAt": "2025-10-26T10:05:00.000Z",
    "latestAnalysisRunId": "run-uuid-1",
    "user": {
      "id": "user-uuid",
      "email": "user@example.com"
    },
    "analysisRuns": [
      {
        "id": "run-uuid-1",
        "runNumber": 1,
        "status": "ANALYZED",
        "createdAt": "2025-10-26T10:00:00.000Z",
        "analyses": [
          {
            "id": "analysis-uuid-1",
            "modelName": "SIGLIP-LSTM-V4",
            "prediction": "FAKE",
            "confidence": 0.92,
            "status": "COMPLETED",
            "processingTime": 45.2,
            "mediaType": "video",
            "resultPayload": {
              "model_name": "SIGLIP-LSTM-V4",
              "prediction": "FAKE",
              "confidence": 0.92,
              "processing_time": 45.2,
              "metadata": {
                "frames_analyzed": 32,
                "rolling_windows": 8
              }
            },
            "createdAt": "2025-10-26T10:00:00.000Z",
            "updatedAt": "2025-10-26T10:01:00.000Z"
          }
        ]
      }
    ]
  }
}
```

**Errors**:

- `400`: Invalid UUID format
- `404`: Media not found or user doesn't have permission

---

### PATCH /media/:id

Update media description.

**Request**:

```http
PATCH /api/v1/media/abc123-uuid
Authorization: Bearer <token>
Content-Type: application/json

{
  "description": "Updated description"
}
```

**Validation**:

```javascript
mediaUpdateSchema = z.object({
  body: z.object({
    description: z.string().max(500, 'Description cannot exceed 500 characters.').optional(),
  }),
  params: z.object({
    id: z.string().uuid('Invalid media ID format.'),
  }),
});
```

**Response** (200 OK):

```json
{
  "statusCode": 200,
  "success": true,
  "message": "Media updated successfully.",
  "data": {
    "id": "abc123-uuid",
    "description": "Updated description",
    "updatedAt": "2025-10-26T11:00:00.000Z"
  }
}
```

---

### DELETE /media/:id

Delete media file and all associated analysis data.

**Request**:

```http
DELETE /api/v1/media/abc123-uuid
Authorization: Bearer <token>
```

**Response** (200 OK):

```json
{
  "statusCode": 200,
  "success": true,
  "message": "Media deleted successfully.",
  "data": {}
}
```

**Side Effects**:

1. Delete file from storage (local/Cloudinary)
2. Cascade delete all `AnalysisRun` records
3. Cascade delete all `DeepfakeAnalysis` records
4. Remove from database

**Errors**:

- `404`: Media not found or user doesn't have permission
- `500`: Failed to delete from storage

---

### POST /media/:id/analyze

Re-run analysis on existing media (creates new AnalysisRun).

**Request**:

```http
POST /api/v1/media/abc123-uuid/analyze
Authorization: Bearer <token>
```

**Response** (202 Accepted):

```json
{
  "statusCode": 202,
  "success": true,
  "message": "A new analysis run has been successfully queued for this media.",
  "data": {
    "id": "abc123-uuid",
    "filename": "video.mp4",
    "status": "QUEUED",
    "latestAnalysisRunId": "new-run-uuid",
    "analysisRuns": [
      {
        "id": "new-run-uuid",
        "runNumber": 2,
        "status": "QUEUED",
        "createdAt": "2025-10-26T12:00:00.000Z",
        "analyses": []
      },
      {
        "id": "old-run-uuid",
        "runNumber": 1,
        "status": "ANALYZED",
        "analyses": [ /* ... */ ]
      }
    ]
  }
}
```

**Use Cases**:

- Analyze with updated ML models
- Re-analyze after initial failure
- Test different model configurations

---

## Monitoring & Health Endpoints

**Base Path**: `/api/v1/monitoring`  
**Authentication**: All endpoints require valid JWT token

### GET /monitoring/server-status

Get current ML Server health and statistics.

**Request**:

```http
GET /api/v1/monitoring/server-status
Authorization: Bearer <token>
```

**Response** (200 OK):

```json
{
  "statusCode": 200,
  "success": true,
  "message": "Server status retrieved successfully",
  "data": {
    "status": "HEALTHY",
    "uptime": 86400,
    "active_models": [
      "SIGLIP-LSTM-V4",
      "COLOR-CUES-LSTM-V1",
      "EFFICIENTNET-B7-V1"
    ],
    "total_models": 15,
    "device": "cuda",
    "memory_usage": {
      "gpu": {
        "allocated": 4096,
        "reserved": 8192,
        "total": 16384
      }
    },
    "responseTimeMs": 120,
    "cachedAt": "2025-10-26T12:00:00.000Z"
  }
}
```

**Caching**: Results cached in Redis for 60 seconds to reduce load on ML Server.

**Errors**:

- `503`: ML Server unavailable

---

### GET /monitoring/server-history

Get historical server health records.

**Request**:

```http
GET /api/v1/monitoring/server-history?limit=50
Authorization: Bearer <token>
```

**Query Parameters**:

- `limit` (optional): Number of records to return (default: 50)

**Response** (200 OK):

```json
{
  "statusCode": 200,
  "success": true,
  "message": "Server health history retrieved successfully",
  "data": [
    {
      "id": "health-uuid-1",
      "status": "HEALTHY",
      "responseTimeMs": 120,
      "statsPayload": { /* Full stats object */ },
      "checkedAt": "2025-10-26T12:00:00.000Z"
    },
    {
      "id": "health-uuid-2",
      "status": "UNHEALTHY",
      "responseTimeMs": 20000,
      "statsPayload": {
        "status": "UNHEALTHY",
        "errorMessage": "Connection timeout"
      },
      "checkedAt": "2025-10-26T11:55:00.000Z"
    }
  ]
}
```

---

### GET /monitoring/queue-status

Get BullMQ job queue statistics.

**Request**:

```http
GET /api/v1/monitoring/queue-status
Authorization: Bearer <token>
```

**Response** (200 OK):

```json
{
  "statusCode": 200,
  "success": true,
  "message": "Processing queue status retrieved successfully.",
  "data": {
    "pending": 5,
    "active": 2,
    "completed": 120,
    "failed": 3,
    "delayed": 0
  }
}
```

**Queue States**:

- **Pending**: Jobs waiting to be processed
- **Active**: Currently processing
- **Completed**: Successfully finished
- **Failed**: Errored during processing
- **Delayed**: Scheduled for future processing

---

### POST /monitoring/check-stuck-runs

Manually trigger stuck run detection and finalization.

**Request**:

```http
POST /api/v1/monitoring/check-stuck-runs
Authorization: Bearer <token>
```

**Response** (200 OK):

```json
{
  "statusCode": 200,
  "success": true,
  "message": "Checked 15 runs and finalized 2 stuck runs.",
  "data": {
    "checked": 15,
    "finalized": 2,
    "stuck_runs": [
      {
        "id": "run-uuid-1",
        "mediaId": "media-uuid-1",
        "status": "PROCESSING",
        "createdAt": "2025-10-26T10:00:00.000Z",
        "stuck_duration_minutes": 45,
        "action": "marked_as_analyzed"
      }
    ]
  }
}
```

**Logic**: Finds runs in `PROCESSING` state for >30 minutes and finalizes them based on completed analyses.

---

### GET /monitoring/verify-statuses

Verify analysis status consistency across Media and AnalysisRun records.

**Request**:

```http
GET /api/v1/monitoring/verify-statuses
Authorization: Bearer <token>
```

**Response** (200 OK):

```json
{
  "statusCode": 200,
  "success": true,
  "message": "Verified 50 media items: 48 correct, 2 incorrect.",
  "data": {
    "totalMedia": 50,
    "correctMedia": 48,
    "incorrectMedia": 2,
    "issues": [
      {
        "mediaId": "media-uuid-1",
        "currentStatus": "PROCESSING",
        "expectedStatus": "ANALYZED",
        "reason": "Latest run is ANALYZED but media status is PROCESSING"
      }
    ]
  }
}
```

---

### POST /monitoring/fix-statuses

Auto-fix detected status inconsistencies.

**Request**:

```http
POST /api/v1/monitoring/fix-statuses
Authorization: Bearer <token>
```

**Response** (200 OK):

```json
{
  "statusCode": 200,
  "success": true,
  "message": "Fixed 2 status issues, 0 failures.",
  "data": {
    "fixed": 2,
    "failed": 0
  }
}
```

---

## PDF Report Endpoints

**Base Path**: `/api/v1/pdf`  
**Authentication**: All endpoints require valid JWT token

### GET /pdf/report/run/:analysisRunId

Generate and download PDF report for a specific analysis run.

**Request**:

```http
GET /api/v1/pdf/report/run/abc123-run-uuid
Authorization: Bearer <token>
```

**Validation**:

```javascript
analysisRunIdParamSchema = z.object({
  params: z.object({
    analysisRunId: z.string().uuid('Invalid analysis run ID format'),
  }),
});
```

**Response** (200 OK):

```http
HTTP/1.1 200 OK
Content-Type: application/pdf
Content-Disposition: attachment; filename="analysis-report-abc123.pdf"
Content-Length: 524288
Cache-Control: no-cache, no-store, must-revalidate

<binary PDF data>
```

**Report Contents**:

1. **Executive Summary**: Overall prediction, confidence, run metadata
2. **Model Results Table**: All model predictions with confidence scores
3. **Detailed Analysis**: Per-model breakdowns with processing times
4. **Metadata**: Media file info, timestamps, user details
5. **Visualizations**: Confidence distribution charts (if available)

**Process Flow**:

1. Verify user has permission to access the run
2. Fetch AnalysisRun with all DeepfakeAnalysis records
3. Generate markdown report from template
4. Convert markdown to PDF using `md-to-pdf`
5. Stream binary PDF to client

**Errors**:

- `400`: Invalid UUID format
- `404`: Analysis run not found or user doesn't have permission
- `500`: PDF generation failed

---

### GET /pdf/test

Generate test PDF for debugging.

**Request**:

```http
GET /api/v1/pdf/test
Authorization: Bearer <token>
```

**Response** (200 OK):

```http
HTTP/1.1 200 OK
Content-Type: application/pdf
Content-Disposition: attachment; filename="test-report.pdf"

<binary PDF data>
```

**Use Case**: Verify PDF generation pipeline without requiring actual analysis data.

---

## Rate Limiting

### General API Rate Limit

**Applied to**: All `/api/*` routes  
**Window**: 1 minute  
**Max Requests**: 20,000 requests per minute per IP

```javascript
// src/middleware/security.middleware.js
export const apiRateLimiter = createRateLimiter({
  windowMs: 1 * 60 * 1000,  // 1 minute
  max: 20000,                 // 20k requests
});
```

### Login Rate Limit

**Applied to**: `POST /api/v1/auth/login`  
**Window**: 10 minutes  
**Max Requests**: 10 failed login attempts per 10 minutes  
**Skip**: Successful login attempts don't count toward limit

```javascript
export const loginRateLimiter = createRateLimiter({
  windowMs: 10 * 60 * 1000,        // 10 minutes
  max: 10,                           // 10 attempts
  skipSuccessfulRequests: true,    // Only count failures
});
```

**Response** (429 Too Many Requests):

```json
{
  "statusCode": 429,
  "success": false,
  "message": "Too many failed login attempts. Please try again after 10 minutes.",
  "errors": null
}
```

---

## Request Validation

### Validation Strategy

All endpoints use **Zod** for schema validation via custom middleware:

```javascript
// src/api/auth/auth.validation.js
export const validate = (schema) => (req, res, next) => {
  try {
    schema.parse({
      body: req.body,
      query: req.query,
      params: req.params,
    });
    next();
  } catch (err) {
    const validationErrors = err.errors.map(e => e.message);
    next(new ApiError(400, "Validation failed", validationErrors));
  }
};
```

### Example Validation Schemas

#### UUID Parameter Validation

```javascript
mediaIdParamSchema = z.object({
  params: z.object({
    id: z.string().uuid('Invalid media ID format.'),
  }),
});
```

#### Body Field Validation

```javascript
updateProfileSchema = z.object({
  body: z.object({
    firstName: z.string().min(1).optional(),
    lastName: z.string().min(1).optional(),
  }),
});
```

#### Email & Password Validation

```javascript
signupSchema = z.object({
  body: z.object({
    email: z.string().email('Invalid email address'),
    password: z.string().min(6, 'Password must be at least 6 characters long'),
    firstName: z.string().min(1, 'First name is required'),
    lastName: z.string().min(1, 'Last name is required'),
  }),
});
```

### Validation Error Response

```json
{
  "statusCode": 400,
  "success": false,
  "message": "Validation failed",
  "errors": [
    "Invalid email address",
    "Password must be at least 6 characters long"
  ]
}
```

---

## Summary

The Drishtiksha Backend API provides:

✅ **RESTful Design** - Standard HTTP methods (GET/POST/PUT/PATCH/DELETE)  
✅ **JWT Authentication** - Secure token-based auth with 7-day expiration  
✅ **Comprehensive Validation** - Zod schemas for all inputs  
✅ **Rate Limiting** - Prevent abuse with configurable limits  
✅ **Consistent Responses** - Unified `ApiResponse` format  
✅ **Error Handling** - Detailed error messages with proper status codes  
✅ **Async Processing** - BullMQ job queue for long-running tasks  
✅ **Real-time Updates** - WebSocket progress events (see WebSocket documentation)  
✅ **PDF Generation** - Downloadable analysis reports  
✅ **Monitoring Tools** - Health checks, queue status, stuck run detection

**Total Endpoints**: 24 (6 auth + 6 media + 6 monitoring + 2 pdf + 2 health + 2 root)

**Next Steps**:

- [Database Schema & Prisma Documentation](./Database-Schema.md)
- [Services & Business Logic Documentation](./Services.md)
- [Middleware & Authentication Documentation](./Middleware.md)
- [WebSocket & Real-time Updates Documentation](./WebSocket.md)
