# Middleware & Authentication

Comprehensive documentation for middleware components in the **Drishtiksha Backend**. Middleware handles cross-cutting concerns like authentication, validation, error handling, and request processing.

---

## Table of Contents

1. [Middleware Architecture](#middleware-architecture)
2. [Authentication Middleware](#authentication-middleware)
3. [Validation Middleware](#validation-middleware)
4. [File Upload Middleware](#file-upload-middleware)
5. [Error Handling Middleware](#error-handling-middleware)
6. [Rate Limiting Middleware](#rate-limiting-middleware)
7. [CORS Middleware](#cors-middleware)
8. [Request Logging Middleware](#request-logging-middleware)
9. [Middleware Execution Order](#middleware-execution-order)

---

## Middleware Architecture

### What is Middleware?

Middleware functions are functions that have access to the request object (`req`), response object (`res`), and the next middleware function in the application's request-response cycle (`next`).

### Middleware Flow

```text
Request
  │
  ├─► [CORS] ────────► Set CORS headers
  │
  ├─► [Logger] ───────► Log request info
  │
  ├─► [Auth] ─────────► Verify JWT token
  │
  ├─► [Validation] ───► Validate request body/params
  │
  ├─► [Rate Limit] ───► Check request rate
  │
  ├─► [Controller] ───► Handle business logic
  │
  └─► [Error Handler] ► Catch and format errors
```

### Middleware Types

1. **Application-level**: Applied to all routes (`app.use()`)
2. **Router-level**: Applied to specific routers (`router.use()`)
3. **Route-level**: Applied to individual routes

---

## Authentication Middleware

**File**: `src/middleware/auth.middleware.js`

Verifies JWT tokens and attaches user information to requests.

### Implementation

```javascript
import jwt from 'jsonwebtoken';
import { ApiError } from '../utils/ApiError.js';
import { userRepository } from '../repositories/user.repository.js';

export const authenticate = async (req, res, next) => {
  try {
    // 1. Extract token from header
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      throw new ApiError(401, 'No token provided');
    }
    
    const token = authHeader.substring(7); // Remove 'Bearer ' prefix
    
    // 2. Verify token
    let decoded;
    try {
      decoded = jwt.verify(token, process.env.JWT_SECRET);
    } catch (error) {
      if (error.name === 'TokenExpiredError') {
        throw new ApiError(401, 'Token has expired');
      }
      if (error.name === 'JsonWebTokenError') {
        throw new ApiError(401, 'Invalid token');
      }
      throw error;
    }
    
    // 3. Fetch user from database
    const user = await userRepository.findById(decoded.userId);
    if (!user) {
      throw new ApiError(401, 'User no longer exists');
    }
    
    // 4. Attach user to request
    req.user = user;
    next();
    
  } catch (error) {
    next(error);
  }
};
```

### Usage

#### Protecting All Routes in a Router

```javascript
import { Router } from 'express';
import { authenticate } from '../middleware/auth.middleware.js';

const router = Router();

// Apply to all routes in this router
router.use(authenticate);

router.get('/profile', getProfile);
router.put('/profile', updateProfile);
router.delete('/account', deleteAccount);
```

#### Protecting Individual Routes

```javascript
router.get('/public', getPublicData); // No auth required

router.get(
  '/private',
  authenticate, // Auth required
  getPrivateData
);
```

### Token Format

Tokens must be sent in the `Authorization` header:

```text
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Token Payload

```javascript
{
  userId: "123e4567-e89b-12d3-a456-426614174000",
  iat: 1704067200, // Issued at
  exp: 1704672000  // Expires at
}
```

### Error Responses

#### No Token

```json
{
  "success": false,
  "statusCode": 401,
  "message": "No token provided",
  "errors": []
}
```

#### Invalid Token

```json
{
  "success": false,
  "statusCode": 401,
  "message": "Invalid token",
  "errors": []
}
```

#### Expired Token

```json
{
  "success": false,
  "statusCode": 401,
  "message": "Token has expired",
  "errors": []
}
```

---

## Validation Middleware

**File**: `src/middleware/validation.middleware.js`

Validates request data against Zod schemas.

### Architecture

Uses **Zod** for runtime schema validation:

```javascript
import { z } from 'zod';

export const validate = (schema) => (req, res, next) => {
  try {
    // Validate request body against schema
    const validated = schema.parse(req.body);
    
    // Replace body with validated data
    req.body = validated;
    next();
  } catch (error) {
    // Convert Zod errors to ApiError
    if (error instanceof z.ZodError) {
      const errors = error.errors.map((err) => ({
        field: err.path.join('.'),
        message: err.message,
      }));
      
      next(new ApiError(400, 'Validation failed', errors));
    } else {
      next(error);
    }
  }
};
```

### Validation Schemas

#### Signup Schema

**File**: `src/validators/auth.validator.js`

```javascript
import { z } from 'zod';

export const signupSchema = z.object({
  email: z
    .string()
    .email('Invalid email format')
    .min(5, 'Email must be at least 5 characters')
    .max(255, 'Email must not exceed 255 characters'),
  
  password: z
    .string()
    .min(8, 'Password must be at least 8 characters')
    .max(128, 'Password must not exceed 128 characters')
    .regex(
      /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/,
      'Password must contain at least one uppercase letter, one lowercase letter, and one number'
    ),
  
  firstName: z
    .string()
    .min(1, 'First name is required')
    .max(100, 'First name must not exceed 100 characters'),
  
  lastName: z
    .string()
    .min(1, 'Last name is required')
    .max(100, 'Last name must not exceed 100 characters'),
});
```

#### Login Schema

```javascript
export const loginSchema = z.object({
  email: z.string().email('Invalid email format'),
  password: z.string().min(1, 'Password is required'),
});
```

#### Upload Media Schema

```javascript
export const uploadMediaSchema = z.object({
  description: z
    .string()
    .max(500, 'Description must not exceed 500 characters')
    .optional(),
});
```

#### Update Profile Schema

```javascript
export const updateProfileSchema = z.object({
  firstName: z
    .string()
    .min(1, 'First name is required')
    .max(100, 'First name must not exceed 100 characters')
    .optional(),
  
  lastName: z
    .string()
    .min(1, 'Last name is required')
    .max(100, 'Last name must not exceed 100 characters')
    .optional(),
});
```

### Usage

```javascript
import { validate } from '../middleware/validation.middleware.js';
import { signupSchema, loginSchema } from '../validators/auth.validator.js';

router.post('/signup', validate(signupSchema), signup);
router.post('/login', validate(loginSchema), login);
```

### Validation Error Response

```json
{
  "success": false,
  "statusCode": 400,
  "message": "Validation failed",
  "errors": [
    {
      "field": "email",
      "message": "Invalid email format"
    },
    {
      "field": "password",
      "message": "Password must be at least 8 characters"
    }
  ]
}
```

---

## File Upload Middleware

**File**: `src/middleware/upload.middleware.js`

Handles multipart/form-data file uploads using Multer.

### Configuration

```javascript
import multer from 'multer';
import { ApiError } from '../utils/ApiError.js';

// Memory storage (file buffered in RAM)
const storage = multer.memoryStorage();

// File filter
const fileFilter = (req, file, cb) => {
  const allowedMimeTypes = [
    // Videos
    'video/mp4',
    'video/avi',
    'video/mkv',
    'video/mov',
    'video/quicktime',
    
    // Audios
    'audio/mpeg',
    'audio/wav',
    'audio/mp3',
    
    // Images
    'image/jpeg',
    'image/png',
    'image/jpg',
  ];
  
  if (allowedMimeTypes.includes(file.mimetype)) {
    cb(null, true); // Accept file
  } else {
    cb(
      new ApiError(
        400,
        `File type not allowed. Allowed types: ${allowedMimeTypes.join(', ')}`
      ),
      false
    );
  }
};

// File size limit: 50MB
const limits = {
  fileSize: 50 * 1024 * 1024, // 50MB in bytes
};

export const upload = multer({
  storage,
  fileFilter,
  limits,
});
```

### Usage

#### Single File Upload

```javascript
import { upload } from '../middleware/upload.middleware.js';

router.post(
  '/upload',
  authenticate,
  upload.single('file'), // Field name: 'file'
  uploadMedia
);
```

#### Multiple Files

```javascript
router.post(
  '/upload-multiple',
  authenticate,
  upload.array('files', 5), // Max 5 files
  uploadMultipleMedia
);
```

### Accessing Uploaded File

```javascript
async function uploadMedia(req, res, next) {
  const file = req.file;
  
  console.log(file.originalname); // "video.mp4"
  console.log(file.mimetype);     // "video/mp4"
  console.log(file.size);         // 15728640 (bytes)
  console.log(file.buffer);       // <Buffer ...>
}
```

### File Upload Errors

#### No File Uploaded

```json
{
  "success": false,
  "statusCode": 400,
  "message": "No file uploaded",
  "errors": []
}
```

#### File Too Large

```json
{
  "success": false,
  "statusCode": 400,
  "message": "File too large. Maximum size: 50MB",
  "errors": []
}
```

#### Invalid File Type

```json
{
  "success": false,
  "statusCode": 400,
  "message": "File type not allowed. Allowed types: video/mp4, audio/mpeg, image/jpeg, ...",
  "errors": []
}
```

---

## Error Handling Middleware

**File**: `src/middleware/error.middleware.js`

Centralized error handling for the entire application.

### Implementation

```javascript
import { ApiError } from '../utils/ApiError.js';
import { ApiResponse } from '../utils/ApiResponse.js';

export const errorHandler = (err, req, res, next) => {
  // Log error for debugging
  console.error('Error:', err);
  
  // Default to 500 Internal Server Error
  let statusCode = 500;
  let message = 'Internal Server Error';
  let errors = [];
  
  // Handle ApiError instances
  if (err instanceof ApiError) {
    statusCode = err.statusCode;
    message = err.message;
    errors = err.errors || [];
  }
  
  // Handle Prisma errors
  else if (err.name === 'PrismaClientKnownRequestError') {
    statusCode = 400;
    message = 'Database operation failed';
    
    // Unique constraint violation
    if (err.code === 'P2002') {
      const field = err.meta?.target?.[0] || 'field';
      message = `${field} already exists`;
    }
    
    // Foreign key constraint violation
    if (err.code === 'P2003') {
      message = 'Related record not found';
    }
  }
  
  // Handle JWT errors
  else if (err.name === 'JsonWebTokenError') {
    statusCode = 401;
    message = 'Invalid token';
  }
  else if (err.name === 'TokenExpiredError') {
    statusCode = 401;
    message = 'Token has expired';
  }
  
  // Handle Multer errors
  else if (err.name === 'MulterError') {
    statusCode = 400;
    if (err.code === 'LIMIT_FILE_SIZE') {
      message = 'File too large. Maximum size: 50MB';
    } else if (err.code === 'LIMIT_UNEXPECTED_FILE') {
      message = 'Unexpected file field';
    } else {
      message = err.message;
    }
  }
  
  // Send error response
  res.status(statusCode).json(
    new ApiResponse(statusCode, null, message, errors)
  );
};
```

### Error Response Format

All errors follow the same structure:

```json
{
  "success": false,
  "statusCode": 400,
  "message": "Error message",
  "data": null,
  "errors": [
    {
      "field": "fieldName",
      "message": "Field-specific error"
    }
  ]
}
```

### Custom Error Class

**File**: `src/utils/ApiError.js`

```javascript
export class ApiError extends Error {
  constructor(statusCode, message, errors = []) {
    super(message);
    this.statusCode = statusCode;
    this.errors = errors;
    this.name = 'ApiError';
    Error.captureStackTrace(this, this.constructor);
  }
}
```

### Usage in Controllers/Services

```javascript
// Not found
throw new ApiError(404, 'Media not found');

// Unauthorized
throw new ApiError(403, 'You do not have permission to access this resource');

// Validation error with field details
throw new ApiError(400, 'Validation failed', [
  { field: 'email', message: 'Invalid email format' },
  { field: 'password', message: 'Password too short' },
]);
```

---

## Rate Limiting Middleware

**File**: `src/middleware/rate-limit.middleware.js`

Prevents abuse by limiting request rates per IP or user.

### Implementation

```javascript
import rateLimit from 'express-rate-limit';
import RedisStore from 'rate-limit-redis';
import { redis } from '../config/redis.js';

// General API rate limiter
export const apiLimiter = rateLimit({
  store: new RedisStore({
    client: redis,
    prefix: 'rl:api:',
  }),
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Max 100 requests per windowMs
  message: {
    success: false,
    statusCode: 429,
    message: 'Too many requests, please try again later',
    errors: [],
  },
  standardHeaders: true,
  legacyHeaders: false,
});

// Auth rate limiter (stricter)
export const authLimiter = rateLimit({
  store: new RedisStore({
    client: redis,
    prefix: 'rl:auth:',
  }),
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 5, // Max 5 login/signup attempts per 15 minutes
  skipSuccessfulRequests: true, // Don't count successful logins
  message: {
    success: false,
    statusCode: 429,
    message: 'Too many authentication attempts, please try again later',
    errors: [],
  },
});

// Upload rate limiter
export const uploadLimiter = rateLimit({
  store: new RedisStore({
    client: redis,
    prefix: 'rl:upload:',
  }),
  windowMs: 60 * 60 * 1000, // 1 hour
  max: 10, // Max 10 uploads per hour
  message: {
    success: false,
    statusCode: 429,
    message: 'Upload limit reached, please try again later',
    errors: [],
  },
});
```

### Usage

```javascript
import { apiLimiter, authLimiter, uploadLimiter } from '../middleware/rate-limit.middleware.js';

// Apply to all routes
app.use('/api', apiLimiter);

// Apply to auth routes
router.post('/login', authLimiter, login);
router.post('/signup', authLimiter, signup);

// Apply to upload route
router.post('/media/upload', uploadLimiter, uploadMedia);
```

### Rate Limit Response

```json
{
  "success": false,
  "statusCode": 429,
  "message": "Too many requests, please try again later",
  "errors": []
}
```

### Response Headers

```text
RateLimit-Limit: 100
RateLimit-Remaining: 95
RateLimit-Reset: 1704067800
```

---

## CORS Middleware

**File**: `src/app.js`

Handles Cross-Origin Resource Sharing.

### Configuration

```javascript
import cors from 'cors';

const corsOptions = {
  origin: process.env.CORS_ORIGIN || 'http://localhost:5173',
  credentials: true, // Allow cookies
  optionsSuccessStatus: 200,
};

app.use(cors(corsOptions));
```

### Multiple Origins

```javascript
const allowedOrigins = [
  'http://localhost:5173',
  'http://localhost:3000',
  'https://drishtiksha.com',
];

const corsOptions = {
  origin: (origin, callback) => {
    if (!origin || allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  },
  credentials: true,
};
```

### Preflight Requests

CORS automatically handles `OPTIONS` preflight requests:

```text
OPTIONS /api/media/upload HTTP/1.1
Host: localhost:8080
Origin: http://localhost:5173
Access-Control-Request-Method: POST
Access-Control-Request-Headers: authorization, content-type

HTTP/1.1 204 No Content
Access-Control-Allow-Origin: http://localhost:5173
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
Access-Control-Allow-Headers: authorization, content-type
Access-Control-Allow-Credentials: true
```

---

## Request Logging Middleware

**File**: `src/middleware/logger.middleware.js`

Logs all incoming requests for monitoring and debugging.

### Implementation

```javascript
export const requestLogger = (req, res, next) => {
  const start = Date.now();
  
  // Log request
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
  
  // Log response on finish
  res.on('finish', () => {
    const duration = Date.now() - start;
    console.log(
      `[${new Date().toISOString()}] ${req.method} ${req.url} ${res.statusCode} ${duration}ms`
    );
  });
  
  next();
};
```

### Enhanced Logging with Winston

```javascript
import winston from 'winston';

const logger = winston.createLogger({
  level: 'info',
  format: winston.format.json(),
  transports: [
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' }),
  ],
});

export const requestLogger = (req, res, next) => {
  const start = Date.now();
  
  res.on('finish', () => {
    const duration = Date.now() - start;
    
    logger.info({
      method: req.method,
      url: req.url,
      status: res.statusCode,
      duration: `${duration}ms`,
      ip: req.ip,
      userAgent: req.get('user-agent'),
    });
  });
  
  next();
};
```

### Sample Log Output

```text
[2025-01-09T12:00:00.000Z] POST /api/auth/login
[2025-01-09T12:00:00.123Z] POST /api/auth/login 200 123ms

[2025-01-09T12:05:30.000Z] GET /api/media
[2025-01-09T12:05:30.045Z] GET /api/media 200 45ms

[2025-01-09T12:10:15.000Z] POST /api/media/upload
[2025-01-09T12:10:15.456Z] POST /api/media/upload 202 456ms
```

---

## Middleware Execution Order

The order of middleware is **critical**. Middleware executes top-to-bottom.

### Correct Order

```javascript
import express from 'express';
import cors from 'cors';
import { requestLogger } from './middleware/logger.middleware.js';
import { errorHandler } from './middleware/error.middleware.js';
import { apiLimiter } from './middleware/rate-limit.middleware.js';

const app = express();

// 1. CORS (must be early)
app.use(cors(corsOptions));

// 2. Body parsers
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// 3. Request logger
app.use(requestLogger);

// 4. Rate limiter
app.use('/api', apiLimiter);

// 5. Routes
app.use('/api/auth', authRoutes);
app.use('/api/media', mediaRoutes);

// 6. 404 Handler
app.use((req, res) => {
  res.status(404).json(
    new ApiResponse(404, null, 'Route not found')
  );
});

// 7. Error handler (must be last)
app.use(errorHandler);
```

### Why Order Matters

#### CORS First

CORS must run before routes to handle preflight `OPTIONS` requests.

#### Body Parsers Before Routes

Routes need access to `req.body` parsed data.

#### Error Handler Last

Error handler must catch errors from all previous middleware and routes.

### Common Mistakes

❌ **Bad** - Error handler before routes:

```javascript
app.use(errorHandler); // Wrong position
app.use('/api', routes);
```

❌ **Bad** - Auth before body parser:

```javascript
app.use(authenticate); // Can't access req.body yet
app.use(express.json());
```

✅ **Good** - Correct order:

```javascript
app.use(express.json());
app.use(authenticate);
app.use('/api', routes);
app.use(errorHandler);
```

---

## Middleware Chaining Example

Multiple middleware can be chained on a single route:

```javascript
router.post(
  '/media/upload',
  authenticate,           // 1. Verify JWT
  authLimiter,           // 2. Check rate limit
  upload.single('file'), // 3. Parse multipart
  validate(uploadSchema),// 4. Validate body
  uploadMedia            // 5. Controller
);
```

Execution flow:

```text
Request
  │
  ├─► authenticate ────────► req.user = {...}
  │
  ├─► authLimiter ─────────► Check Redis rate limit
  │
  ├─► upload.single('file')► req.file = {...}
  │
  ├─► validate(uploadSchema)► Validate req.body
  │
  └─► uploadMedia ─────────► Business logic
```

---

## Summary

The middleware layer provides:

✅ **Authentication** - JWT verification and user attachment  
✅ **Validation** - Runtime schema validation with Zod  
✅ **File Uploads** - Multipart parsing with Multer  
✅ **Error Handling** - Centralized error formatting  
✅ **Rate Limiting** - Redis-backed request throttling  
✅ **CORS** - Cross-origin request handling  
✅ **Logging** - Request/response monitoring  
✅ **Security** - Protection against common attacks  

**Next Steps**:

- [WebSocket & Real-time Updates](./WebSocket.md)
- [Services & Business Logic](./Services.md)
- [API Routes Reference](./API-Routes.md)
