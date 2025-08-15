# Drishtiksha AI - Backend API Module

## Overview

The backend module is a robust Node.js/Express API server that serves as the central orchestrator for the Drishtiksha AI Deepfake Detection System. It provides secure authentication, video management, analysis coordination with ML models, and comprehensive data persistence using PostgreSQL with Prisma ORM.

## Tech Stack

### Core Technologies

- Node.js 18+ - JavaScript runtime for server-side development
- Express 5.1.0 - Fast, unopinionated web framework for Node.js
- Prisma 6.14.0 - Next-generation ORM for type-safe database access
- PostgreSQL - Robust relational database for data persistence
- JWT - JSON Web Tokens for secure authentication

### Media & File Processing

- Multer 1.4.5 - Middleware for handling multipart/form-data (file uploads)
- Cloudinary 3.0.1 - Cloud-based image and video management
- FFmpeg - Video processing and manipulation (via system binaries)
- Sharp - High-performance image processing

### Communication & Integration

- Axios 1.9.0 - Promise-based HTTP client for ML server communication
- Bull 4.16.5 - Redis-based queue system for background processing
- Socket.io 5.1.2 - Real-time bidirectional event-based communication
- Redis 5.2.1 - In-memory data store for caching and queues

### Development & Testing

- Jest 30.0.0 - JavaScript testing framework
- Supertest 7.0.0 - HTTP assertion library for testing APIs
- Nodemon 3.2.0 - Development utility for auto-restarting server
- ESLint - Code linting and quality assurance

### Security & Validation

- bcryptjs 2.4.3 - Password hashing library
- joi 18.0.1 - Object schema validation
- helmet 8.0.0 - Express middleware for security headers
- cors 2.8.5 - Cross-Origin Resource Sharing middleware

## Project Structure

```bash
Backend/
├── src/
│ ├── app.js        # Express application setup
│ ├── api/        # API route definitions
│ │ ├── auth/       # Authentication endpoints
│ │ │ ├── auth.routes.js  # Login, register, refresh routes
│ │ │ └── auth.controller.js # Authentication business logic
│ │ ├── health/     # Health check endpoints
│ │ │ ├── health.routes.js  # System health monitoring
│ │ │ └── health.controller.js # Health check logic
│ │ └── videos/     # Video management endpoints
│ │   ├── videos.routes.js  # Video CRUD and analysis routes
│ │   └── videos.controller.js # Video business logic
│ ├── config/       # Configuration files
│ │ └── database.js     # Database connection and configuration
│ ├── middleware/     # Express middleware
│ │ ├── auth.middleware.js  # JWT authentication middleware
│ │ ├── error.middleware.js # Global error handling
│ │ └── multer.middleware.js  # File upload configuration
│ ├── repositories/     # Data access layer
│ │ ├── user.repository.js  # User database operations
│ │ └── video.repository.js # Video database operations
│ ├── services/       # Business logic layer
│ │ ├── auth.service.js   # Authentication business logic
│ │ ├── video.service.js  # Video processing coordination
│ │ └── modelAnalysis.service.js # ML model integration
│ ├── utils/        # Utility functions and helpers
│ │ ├── ApiError.js     # Custom error class
│ │ ├── ApiResponse.js    # Standardized API responses
│ │ ├── asyncHandler.js   # Async function wrapper
│ │ ├── cloudinary.js   # Cloudinary configuration
│ │ ├── jwt.js      # JWT utilities
│ │ ├── logger.js     # Application logging
│ │ └── password.js     # Password utilities
│ └── queue/        # Background job processing
│   └── videoProcessorQueue.js # Video analysis queue management
├── prisma/         # Database schema and migrations
│ ├── schema.prisma     # Database schema definition
│ └── migrations/     # Database migration files
├── tests/        # Test suites
│ ├── basic.test.js     # Basic API functionality tests
│ ├── integration.test.js   # Integration tests
│ ├── video-endpoints.test.js # Video API endpoint tests
│ ├── fixtures/       # Test data and fixtures
│ └── setup/        # Test environment setup
│   └── testSetup.js    # Jest configuration and setup
├── temp/         # Temporary file storage
├── uploads/        # File upload storage
│ ├── videos/       # Uploaded video files
│ └── visualizations/     # Generated visualization files
├── server.js       # Application entry point
├── package.json      # Dependencies and scripts
└── jest.config.json      # Jest testing configuration
```

## Key Features

### 🔐 Authentication System

- JWT-based Authentication: Secure token-based authentication
- Password Security: bcrypt hashing with configurable salt rounds
- Token Management: Access tokens with refresh token support
- Role-based Access: User role management and permissions

### 📹 Video Management

- Multi-format Support: MP4, AVI, MOV, and other common video formats
- Cloud Storage: Cloudinary integration for scalable video hosting
- Metadata Extraction: Automatic video metadata processing
- Thumbnail Generation: Automatic video thumbnail creation

### 🤖 ML Model Integration

- Multi-Model Support: Integration with 3 specialized deepfake detection models:
  - SIGLIP-LSTM-V1: Advanced visual analysis
  - SIGLIP-LSTM-V3: Enhanced feature detection
  - ColorCues-LSTM-V1: Color-based deepfake detection
- Analysis Types: Quick, Detailed, Frame-by-Frame, and Visualization modes
- Version Management: Track multiple analysis versions per video/model
- Real-time Processing: Background job processing with progress tracking

### 📊 Data Management

- Prisma ORM: Type-safe database operations with auto-generated client
- PostgreSQL: Robust relational database with ACID compliance
- Migration System: Database version control and schema evolution
- Data Validation: Comprehensive input validation with Joi schemas

### 🚀 Performance & Scalability

- Background Processing: Redis-based queue system for long-running tasks
- Caching Strategy: Redis caching for frequently accessed data
- Connection Pooling: Optimized database connections
- Error Handling: Comprehensive error handling and logging

## Database Schema

### Core Tables

Users Table:

```sql
model User {
  id    String @id @default(cuid())
  email   String @unique
  password  String
  name  String?
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt
  videos  Video[]
}
```

Videos Table:

```sql
model Video {
  id      String @id @default(cuid())
  filename    String
  originalName  String
  cloudinaryUrl String
  cloudinaryId  String
  metadata    Json?
  uploadedAt  DateTime @default(now())
  userId    String
  user    User   @relation(fields: [userId], references: [id])
  analyses    Analysis[]
}
```

Analysis Table:

```sql
model Analysis {
  id      String @id @default(cuid())
  videoId   String
  model     String // SIGLIP_LSTM_V1, SIGLIP_LSTM_V3, COLOR_CUES_LSTM_V1
  analysisType  String // QUICK, DETAILED, FRAMES, VISUALIZE
  version   Int
  status    String @default("PENDING")
  result    Json?
  error     String?
  startedAt   DateTime @default(now())
  completedAt   DateTime?
  video     Video  @relation(fields: [videoId], references: [id])

  @@unique([videoId, model, analysisType, version])
}
```

## Environment Configuration

### Required Environment Variables

Create a `.env` file in the Backend directory:

```env
# Database Configuration
DATABASE_URL="postgresql://username:password@localhost:5432/drishtiksha_db"

# JWT Configuration
JWT_SECRET="your-super-secret-jwt-key-here"
JWT_REFRESH_SECRET="your-super-secret-refresh-key-here"
JWT_EXPIRES_IN="1h"
JWT_REFRESH_EXPIRES_IN="7d"

# Cloudinary Configuration
CLOUDINARY_CLOUD_NAME="your-cloud-name"
CLOUDINARY_API_KEY="your-api-key"
CLOUDINARY_API_SECRET="your-api-secret"

# ML Server Configuration
ML_SERVER_URL="http://localhost:8000"
ML_SERVER_API_KEY="your-ml-server-api-key"

# Redis Configuration
REDIS_URL="redis://localhost:6379"

# Application Configuration
PORT=3000
NODE_ENV="development"
CORS_ORIGIN="http://localhost:5173"

# File Upload Configuration
MAX_FILE_SIZE="100MB"
ALLOWED_FILE_TYPES="video/mp4,video/avi,video/mov,video/webm"

# Queue Configuration
QUEUE_REDIS_URL="redis://localhost:6379"
QUEUE_CONCURRENCY=5
```

### Environment-Specific Settings

Development:

- Detailed error messages and stack traces
- Request/response logging
- Hot reload with nodemon
- CORS enabled for frontend development

Production:

- Optimized error handling
- Security headers with helmet
- Rate limiting
- Compressed responses

## Installation & Setup

### Prerequisites

- Node.js 18+ and npm/yarn
- PostgreSQL 12+ database server
- Redis server for queues and caching
- Cloudinary account for media storage
- ML Server running on port 8000

### Quick Start

```bash
# Navigate to backend directory
cd Backend

# Install dependencies
npm install

# Setup environment variables
cp .env.example .env
# Edit .env with your configuration

# Initialize database
npx prisma generate
npx prisma db push

# Run database migrations
npx prisma migrate dev

# Start development server
npm run dev

# Alternative: Start with PM2 for production
npm run start:pm2
```

### Database Setup

```bash
# Generate Prisma client
npx prisma generate

# Run migrations
npx prisma migrate dev --name init

# Seed database (if seeder exists)
npx prisma db seed

# View database in Prisma Studio
npx prisma studio
```

## API Documentation

### Authentication Endpoints

#### POST /api/auth/register

```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "password": "securePassword123"
}
```

#### POST /api/auth/login

```json
{
  "email": "john@example.com",
  "password": "securePassword123"
}
```

#### POST /api/auth/refresh

```json
{
  "refreshToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### Video Management Endpoints

#### POST /api/videos/upload

- Content-Type: multipart/form-data
- Field: video (file)
- Returns: Video metadata and Cloudinary URL

#### GET /api/videos

- Returns: List of user's uploaded videos

#### GET /api/videos/:id

- Returns: Specific video details with analysis history

#### DELETE /api/videos/:id

- Deletes video and all associated analyses

### Analysis Endpoints

#### POST /api/videos/:id/analyze

```json
{
  "model": "SIGLIP_LSTM_V1",
  "analysisType": "DETAILED"
}
```

#### GET /api/videos/:id/analyses

- Returns: All analyses for a specific video

#### GET /api/analyses/:id

- Returns: Detailed analysis results

### Health Check Endpoints

#### GET /api/health

- Returns: System health status

#### GET /api/health/detailed

- Returns: Detailed system diagnostics

## Service Layer Architecture

### Video Service

```javascript
class VideoService {
  async uploadVideo(file, userId) {
    // Upload to Cloudinary
    // Save metadata to database
    // Return video record
  }

  async triggerAnalysis(videoId, model, analysisType) {
    // Validate parameters
    // Create analysis record
    // Queue background job
    // Return analysis ID
  }

  async getAnalysisResults(analysisId) {
    // Retrieve from database
    // Format response
    // Return structured data
  }
}
```

### ML Integration Service

```javascript
class ModelAnalysisService {
  async analyzeVideo(videoUrl, model, analysisType) {
    // Prepare ML server request
    // Send to appropriate model endpoint
    // Handle response and errors
    // Return analysis results
  }

  async getModelStatus() {
    // Check ML server health
    // Verify model availability
    // Return status information
  }
}
```

## Background Job Processing

### Queue System

```javascript
// Video Processing Queue
const videoQueue = new Bull("video processing", {
  redis: { host: "localhost", port: 6379 },
});

videoQueue.process("analyze-video", async (job) => {
  const { videoId, model, analysisType } = job.data;

  try {
    // Update status to PROCESSING
    await updateAnalysisStatus(job.data.analysisId, "PROCESSING");

    // Process with ML server
    const result = await processWithMLServer(videoId, model, analysisType);

    // Save results
    await saveAnalysisResults(job.data.analysisId, result);

    // Update status to COMPLETED
    await updateAnalysisStatus(job.data.analysisId, "COMPLETED");
  } catch (error) {
    // Handle errors and update status
    await updateAnalysisStatus(
    job.data.analysisId,
    "FAILED",
    error.message
    );
    throw error;
  }
});
```

## Testing

### Test Suites

Unit Tests:

```bash
# Run all tests
npm run test

# Run specific test file
npm run test -- tests/video-endpoints.test.js

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage
```

Integration Tests:

```bash
# Run integration tests
npm run test:integration

# Test specific endpoints
npm run test -- --grep "video upload"
```

### Test Configuration

```javascript
// jest.config.json
{
  "testEnvironment": "node",
  "setupFilesAfterEnv": ["<rootDir>/tests/setup/testSetup.js"],
  "collectCoverageFrom": [
  "src//*.js",
  "!src//*.test.js"
  ],
  "coverageDirectory": "coverage",
  "coverageReporters": ["text", "lcov", "html"]
}
```

## Security Implementation

### Authentication Middleware

```javascript
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers["authorization"];
  const token = authHeader && authHeader.split(" ")[1];

  if (!token) {
    return res.status(401).json({ error: "Access token required" });
  }

  jwt.verify(token, process.env.JWT_SECRET, (err, user) => {
    if (err) return res.status(403).json({ error: "Invalid token" });
    req.user = user;
    next();
  });
};
```

### Input Validation

```javascript
const videoUploadSchema = Joi.object({
  analysisType: Joi.string()
    .valid("QUICK", "DETAILED", "FRAMES", "VISUALIZE")
    .required(),
  model: Joi.string()
    .valid("SIGLIP_LSTM_V1", "SIGLIP_LSTM_V3", "COLOR_CUES_LSTM_V1")
    .required(),
});
```

### Security Headers

```javascript
app.use(
  helmet({
    contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      scriptSrc: ["'self'"],
      imgSrc: ["'self'", "data:", "https:"],
    },
    },
  })
);
```

## Performance Optimization

### Caching Strategy

```javascript
// Redis caching for frequently accessed data
const getVideoAnalyses = async (videoId) => {
  const cacheKey = `video:${videoId}:analyses`;
  const cached = await redis.get(cacheKey);

  if (cached) {
    return JSON.parse(cached);
  }

  const analyses = await database.analysis.findMany({
    where: { videoId },
  });

  await redis.setex(cacheKey, 300, JSON.stringify(analyses)); // 5 min cache
  return analyses;
};
```

### Database Optimization

```javascript
// Optimized queries with Prisma
const getVideoWithAnalyses = async (videoId) => {
  return await prisma.video.findUnique({
    where: { id: videoId },
    include: {
    analyses: {
      orderBy: { createdAt: "desc" },
      take: 10, // Limit recent analyses
    },
    },
  });
};
```

## Deployment

### Production Build

```bash
# Install production dependencies only
npm ci --only=production

# Run database migrations
npx prisma migrate deploy

# Start with PM2
npm run start:pm2

# Monitor processes
pm2 status
pm2 logs drishtiksha-api
```

### Docker Deployment

```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

RUN npx prisma generate

EXPOSE 3000

CMD ["npm", "start"]
```

### Environment-specific Configurations

Production:

- Enable compression middleware
- Set up rate limiting
- Configure proper logging
- Enable security headers
- Set up monitoring and alerts

## Monitoring & Logging

### Application Logging

```javascript
const winston = require("winston");

const logger = winston.createLogger({
  level: "info",
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({
    filename: "logs/error.log",
    level: "error",
    }),
    new winston.transports.File({ filename: "logs/combined.log" }),
    new winston.transports.Console({
    format: winston.format.simple(),
    }),
  ],
});
```

### Health Monitoring

```javascript
// Health check endpoint
app.get("/api/health", async (req, res) => {
  const health = {
    status: "OK",
    timestamp: new Date().toISOString(),
    services: {
    database: await checkDatabaseConnection(),
    redis: await checkRedisConnection(),
    mlServer: await checkMLServerConnection(),
    },
  };

  const isHealthy = Object.values(health.services).every(
    (service) => service === "OK"
  );
  res.status(isHealthy ? 200 : 503).json(health);
});
```

## Troubleshooting

### Common Issues

Database Connection Errors:

- Verify PostgreSQL is running
- Check DATABASE_URL format
- Ensure database exists and user has permissions

File Upload Issues:

- Check file size limits
- Verify Cloudinary configuration
- Ensure sufficient disk space for temporary files

ML Server Communication:

- Verify ML server is running on port 8000
- Check network connectivity
- Validate API key configuration

Queue Processing Issues:

- Ensure Redis server is running
- Check queue configuration
- Monitor queue status with Bull dashboard

### Debug Mode

```bash
# Enable debug logging
DEBUG=* npm run dev

# Database query logging
DATABASE_LOGGING=true npm run dev

# Queue debugging
QUEUE_DEBUG=true npm run dev
```

## Contributing

### Development Guidelines

1. Follow RESTful API design principles
2. Implement proper error handling and logging
3. Write comprehensive tests for new features
4. Use Prisma for all database operations
5. Follow security best practices

### Code Style

- Use ESLint for code linting
- Follow Airbnb JavaScript style guide
- Use async/await for asynchronous operations
- Implement proper error handling with try-catch
- Use meaningful variable and function names

## License

This project is part of the Drishtiksha AI Deepfake Detection System and is proprietary software developed for educational and research purposes.

---

Last Updated: August 15, 2025  
Version: 1.0.0  
Maintainer: Drishtiksha AI Team
