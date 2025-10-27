# WebSocket & Real-time Updates

Comprehensive documentation for the **Drishtiksha Backend** real-time communication system using Socket.IO and Redis Pub/Sub.

---

## Table of Contents

1. [Real-time Architecture Overview](#real-time-architecture-overview)
2. [Socket.IO Server Setup](#socketio-server-setup)
3. [Redis Pub/Sub Integration](#redis-pubsub-integration)
4. [Event Types](#event-types)
5. [Client Connection Flow](#client-connection-flow)
6. [Authentication & Authorization](#authentication--authorization)
7. [Event Publishing from Workers](#event-publishing-from-workers)
8. [Client SDK Usage](#client-sdk-usage)
9. [Error Handling & Reconnection](#error-handling--reconnection)
10. [Performance & Scaling](#performance--scaling)

---

## Real-time Architecture Overview

### Why WebSockets?

Deepfake analysis can take minutes to complete. Without real-time updates, users would need to manually refresh to see progress. WebSockets enable **server-push notifications** for:

- Media upload confirmation
- Analysis job queue status
- Individual model progress updates
- Analysis completion/failure notifications
- Overall run status changes

### Architecture Components

```text
┌────────────────────────────────────────────────────────────┐
│                      Drishtiksha System                     │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐         ┌──────────────┐                │
│  │  Frontend    │◄───────►│  API Server  │                │
│  │  (React)     │  HTTP   │  (Express)   │                │
│  └──────┬───────┘         └──────┬───────┘                │
│         │                        │                          │
│         │ WebSocket              │ Publish                  │
│         │ (Socket.IO)            │                          │
│         ▼                        ▼                          │
│  ┌──────────────┐         ┌──────────────┐                │
│  │  Socket.IO   │◄───────►│    Redis     │                │
│  │   Server     │Subscribe│  Pub/Sub     │                │
│  └──────────────┘         └──────┬───────┘                │
│                                   ▲                          │
│                                   │ Publish                 │
│                                   │                          │
│                            ┌──────┴───────┐                │
│                            │  BullMQ      │                │
│                            │  Workers     │                │
│                            └──────────────┘                │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

### Message Flow

1. **API Server** creates media and queues analysis jobs
2. **Worker** processes job and publishes progress to **Redis**
3. **Socket.IO Server** subscribes to Redis and receives events
4. **Socket.IO Server** emits events to connected **clients** (users)
5. **Frontend** receives real-time updates and updates UI

---

## Socket.IO Server Setup

**File**: `src/config/socket.js`

### Server Initialization

```javascript
import { Server } from 'socket.io';
import { createAdapter } from '@socket.io/redis-adapter';
import { redis } from './redis.js';
import jwt from 'jsonwebtoken';
import { ApiError } from '../utils/ApiError.js';

export function initializeSocketServer(httpServer) {
  const io = new Server(httpServer, {
    cors: {
      origin: process.env.CORS_ORIGIN || 'http://localhost:5173',
      credentials: true,
    },
    transports: ['websocket', 'polling'],
  });
  
  // Use Redis adapter for horizontal scaling
  const pubClient = redis.duplicate();
  const subClient = redis.duplicate();
  
  await Promise.all([pubClient.connect(), subClient.connect()]);
  
  io.adapter(createAdapter(pubClient, subClient));
  
  // Authentication middleware
  io.use(async (socket, next) => {
    try {
      const token = socket.handshake.auth.token;
      
      if (!token) {
        throw new ApiError(401, 'Authentication required');
      }
      
      const decoded = jwt.verify(token, process.env.JWT_SECRET);
      socket.userId = decoded.userId;
      
      next();
    } catch (error) {
      next(new Error('Authentication failed'));
    }
  });
  
  // Connection handler
  io.on('connection', (socket) => {
    console.log(`User connected: ${socket.userId}`);
    
    // Join user-specific room
    socket.join(`user:${socket.userId}`);
    
    socket.on('disconnect', () => {
      console.log(`User disconnected: ${socket.userId}`);
    });
  });
  
  return io;
}
```

### Integration with Express Server

**File**: `src/app.js`

```javascript
import express from 'express';
import { createServer } from 'http';
import { initializeSocketServer } from './config/socket.js';

const app = express();

// ... middleware and routes ...

// Create HTTP server
const httpServer = createServer(app);

// Initialize Socket.IO
const io = await initializeSocketServer(httpServer);

// Make io available globally
app.set('io', io);

// Start server
httpServer.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```

---

## Redis Pub/Sub Integration

### Subscriber Setup

**File**: `src/config/socket.js`

```javascript
import { redis } from './redis.js';

export function setupEventSubscriber(io) {
  const subscriber = redis.duplicate();
  await subscriber.connect();
  
  // Subscribe to media progress channel
  await subscriber.subscribe('media-progress-events', (message) => {
    try {
      const event = JSON.parse(message);
      handleEvent(io, event);
    } catch (error) {
      console.error('Failed to parse event:', error);
    }
  });
  
  console.log('Subscribed to media-progress-events channel');
}

function handleEvent(io, event) {
  const { event: eventName, data } = event;
  
  switch (eventName) {
    case 'media_uploaded':
      io.to(`user:${data.userId}`).emit('media_uploaded', data);
      break;
    
    case 'analysis_started':
      io.to(`user:${data.userId}`).emit('analysis_started', data);
      break;
    
    case 'analysis_progress':
      io.to(`user:${data.userId}`).emit('analysis_progress', data);
      break;
    
    case 'analysis_completed':
      io.to(`user:${data.userId}`).emit('analysis_completed', data);
      break;
    
    case 'analysis_failed':
      io.to(`user:${data.userId}`).emit('analysis_failed', data);
      break;
    
    case 'run_finalized':
      io.to(`user:${data.userId}`).emit('run_finalized', data);
      break;
    
    default:
      console.warn(`Unknown event type: ${eventName}`);
  }
}
```

### Publisher Setup

**File**: `src/services/event-publisher.service.js`

```javascript
import { redis } from '../config/redis.js';

class EventPublisher {
  async publish(eventName, data) {
    const payload = JSON.stringify({
      event: eventName,
      data,
      timestamp: new Date().toISOString(),
    });
    
    await redis.publish('media-progress-events', payload);
  }
}

export const eventPublisher = new EventPublisher();
```

---

## Event Types

### `media_uploaded`

**Triggered**: When user uploads new media

**Payload**:

```javascript
{
  userId: "123e4567-e89b-12d3-a456-426614174000",
  mediaId: "987fcdeb-51a2-43c7-9def-123456789abc",
  filename: "video.mp4",
  status: "QUEUED"
}
```

**Client Effect**: Add new media item to UI with "Queued" status

---

### `analysis_started`

**Triggered**: When worker begins processing a model analysis

**Payload**:

```javascript
{
  userId: "123e4567-e89b-12d3-a456-426614174000",
  mediaId: "987fcdeb-51a2-43c7-9def-123456789abc",
  analysisRunId: "abc12345-6789-4def-ghi0-jklmnopqrstu",
  modelName: "SIGLIP-LSTM-V4"
}
```

**Client Effect**: Update UI to show model is processing

---

### `analysis_progress`

**Triggered**: During long-running analysis (optional)

**Payload**:

```javascript
{
  userId: "123e4567-e89b-12d3-a456-426614174000",
  mediaId: "987fcdeb-51a2-43c7-9def-123456789abc",
  analysisRunId: "abc12345-6789-4def-ghi0-jklmnopqrstu",
  modelName: "SIGLIP-LSTM-V4",
  progress: 65 // Percentage
}
```

**Client Effect**: Update progress bar

---

### `analysis_completed`

**Triggered**: When single model analysis finishes successfully

**Payload**:

```javascript
{
  userId: "123e4567-e89b-12d3-a456-426614174000",
  mediaId: "987fcdeb-51a2-43c7-9def-123456789abc",
  analysisRunId: "abc12345-6789-4def-ghi0-jklmnopqrstu",
  analysisId: "def45678-9012-3abc-4567-890abcdef123",
  modelName: "SIGLIP-LSTM-V4",
  prediction: "DEEPFAKE",
  confidence: 0.92
}
```

**Client Effect**: Display result for this model

---

### `analysis_failed`

**Triggered**: When model analysis encounters error

**Payload**:

```javascript
{
  userId: "123e4567-e89b-12d3-a456-426614174000",
  mediaId: "987fcdeb-51a2-43c7-9def-123456789abc",
  analysisRunId: "abc12345-6789-4def-ghi0-jklmnopqrstu",
  modelName: "SIGLIP-LSTM-V4",
  error: "ML server timeout after 30s"
}
```

**Client Effect**: Show error state for this model

---

### `run_finalized`

**Triggered**: After all models complete/fail for an analysis run

**Payload**:

```javascript
{
  userId: "123e4567-e89b-12d3-a456-426614174000",
  mediaId: "987fcdeb-51a2-43c7-9def-123456789abc",
  analysisRunId: "abc12345-6789-4def-ghi0-jklmnopqrstu",
  finalStatus: "ANALYZED", // or "PARTIALLY_ANALYZED", "FAILED"
  totalModels: 5,
  completedCount: 5,
  failedCount: 0
}
```

**Client Effect**: Update overall status, enable download/re-analyze buttons

---

## Client Connection Flow

### Step 1: Authenticate

Client must obtain JWT token first via `/api/auth/login`:

```javascript
const response = await fetch('http://localhost:8080/api/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ email, password }),
});

const { token } = await response.json();
```

### Step 2: Connect to Socket.IO

```javascript
import { io } from 'socket.io-client';

const socket = io('http://localhost:8080', {
  auth: {
    token: token, // JWT token from login
  },
  transports: ['websocket', 'polling'],
});
```

### Step 3: Listen for Events

```javascript
socket.on('connect', () => {
  console.log('Connected to server');
});

socket.on('media_uploaded', (data) => {
  console.log('New media uploaded:', data);
  // Update UI
});

socket.on('analysis_completed', (data) => {
  console.log('Analysis completed:', data);
  // Update UI with results
});

socket.on('disconnect', () => {
  console.log('Disconnected from server');
});
```

---

## Authentication & Authorization

### JWT Verification

Socket.IO middleware verifies JWT token before allowing connection:

```javascript
io.use(async (socket, next) => {
  try {
    const token = socket.handshake.auth.token;
    
    if (!token) {
      return next(new Error('Authentication required'));
    }
    
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    
    // Attach userId to socket
    socket.userId = decoded.userId;
    
    next();
  } catch (error) {
    next(new Error('Invalid token'));
  }
});
```

### Room-Based Authorization

Users are automatically joined to their user-specific room:

```javascript
io.on('connection', (socket) => {
  // Join room: user:<userId>
  socket.join(`user:${socket.userId}`);
});
```

Events are only sent to the owning user:

```javascript
// Only send to this specific user
io.to(`user:${data.userId}`).emit('analysis_completed', data);
```

### Rejected Connection

If authentication fails:

```javascript
socket.on('connect_error', (error) => {
  console.error('Connection failed:', error.message);
  // "Authentication failed"
});
```

---

## Event Publishing from Workers

### Worker Job Handler

**File**: `src/workers/media-analysis.worker.js`

```javascript
import { eventPublisher } from '../services/event-publisher.service.js';

worker.on('run-single-analysis', async (job) => {
  const { mediaId, analysisRunId, modelName, userId } = job.data;
  
  // Publish start event
  await eventPublisher.publish('analysis_started', {
    userId,
    mediaId,
    analysisRunId,
    modelName,
  });
  
  try {
    // Run analysis
    const result = await analysisService.runSingleModelAnalysis(
      mediaId,
      analysisRunId,
      modelName
    );
    
    // Publish completion event
    await eventPublisher.publish('analysis_completed', {
      userId,
      mediaId,
      analysisRunId,
      analysisId: result.id,
      modelName,
      prediction: result.prediction,
      confidence: result.confidence,
    });
  } catch (error) {
    // Publish failure event
    await eventPublisher.publish('analysis_failed', {
      userId,
      mediaId,
      analysisRunId,
      modelName,
      error: error.message,
    });
    
    throw error;
  }
});

worker.on('finalize-analysis-run', async (job) => {
  const { mediaId, analysisRunId, userId } = job.data;
  
  const run = await analysisService.finalizeAnalysisRun(analysisRunId);
  
  // Publish finalization event
  await eventPublisher.publish('run_finalized', {
    userId,
    mediaId,
    analysisRunId,
    finalStatus: run.status,
    totalModels: run.totalModels,
    completedCount: run.completedCount,
    failedCount: run.failedCount,
  });
});
```

---

## Client SDK Usage

### React Integration

**File**: `src/contexts/SocketContext.jsx`

```javascript
import { createContext, useContext, useEffect, useState } from 'react';
import { io } from 'socket.io-client';
import { useAuth } from './AuthContext';

const SocketContext = createContext(null);

export function SocketProvider({ children }) {
  const { token } = useAuth();
  const [socket, setSocket] = useState(null);
  const [connected, setConnected] = useState(false);
  
  useEffect(() => {
    if (!token) return;
    
    const newSocket = io('http://localhost:8080', {
      auth: { token },
      transports: ['websocket', 'polling'],
    });
    
    newSocket.on('connect', () => {
      console.log('Socket connected');
      setConnected(true);
    });
    
    newSocket.on('disconnect', () => {
      console.log('Socket disconnected');
      setConnected(false);
    });
    
    setSocket(newSocket);
    
    return () => {
      newSocket.disconnect();
    };
  }, [token]);
  
  return (
    <SocketContext.Provider value={{ socket, connected }}>
      {children}
    </SocketContext.Provider>
  );
}

export const useSocket = () => useContext(SocketContext);
```

### Using in Components

```javascript
import { useSocket } from '../contexts/SocketContext';
import { useState, useEffect } from 'react';

function MediaList() {
  const { socket } = useSocket();
  const [mediaList, setMediaList] = useState([]);
  
  useEffect(() => {
    if (!socket) return;
    
    socket.on('media_uploaded', (data) => {
      setMediaList(prev => [data, ...prev]);
    });
    
    socket.on('analysis_completed', (data) => {
      setMediaList(prev =>
        prev.map(media =>
          media.id === data.mediaId
            ? { ...media, status: 'PROCESSING' }
            : media
        )
      );
    });
    
    socket.on('run_finalized', (data) => {
      setMediaList(prev =>
        prev.map(media =>
          media.id === data.mediaId
            ? { ...media, status: data.finalStatus }
            : media
        )
      );
    });
    
    return () => {
      socket.off('media_uploaded');
      socket.off('analysis_completed');
      socket.off('run_finalized');
    };
  }, [socket]);
  
  return (
    <div>
      {mediaList.map(media => (
        <MediaCard key={media.id} media={media} />
      ))}
    </div>
  );
}
```

---

## Error Handling & Reconnection

### Automatic Reconnection

Socket.IO automatically reconnects on disconnect:

```javascript
const socket = io('http://localhost:8080', {
  auth: { token },
  reconnection: true,
  reconnectionAttempts: Infinity,
  reconnectionDelay: 1000,
  reconnectionDelayMax: 5000,
});

socket.on('reconnect_attempt', (attemptNumber) => {
  console.log(`Reconnection attempt ${attemptNumber}`);
});

socket.on('reconnect', (attemptNumber) => {
  console.log(`Reconnected after ${attemptNumber} attempts`);
});

socket.on('reconnect_error', (error) => {
  console.error('Reconnection error:', error);
});

socket.on('reconnect_failed', () => {
  console.error('Reconnection failed after all attempts');
});
```

### Connection State Management

```javascript
function ConnectionStatus() {
  const { socket, connected } = useSocket();
  const [status, setStatus] = useState('connecting');
  
  useEffect(() => {
    if (!socket) return;
    
    socket.on('connect', () => setStatus('connected'));
    socket.on('disconnect', () => setStatus('disconnected'));
    socket.on('reconnecting', () => setStatus('reconnecting'));
    
    return () => {
      socket.off('connect');
      socket.off('disconnect');
      socket.off('reconnecting');
    };
  }, [socket]);
  
  return (
    <div className={`status status-${status}`}>
      {status === 'connected' && '🟢 Connected'}
      {status === 'disconnected' && '🔴 Disconnected'}
      {status === 'reconnecting' && '🟡 Reconnecting...'}
    </div>
  );
}
```

### Handling Stale Data

When reconnecting, fetch latest data:

```javascript
socket.on('reconnect', async () => {
  console.log('Reconnected, refreshing data...');
  
  // Fetch latest media list
  const response = await fetch('/api/media', {
    headers: { Authorization: `Bearer ${token}` },
  });
  
  const mediaList = await response.json();
  setMediaList(mediaList);
});
```

---

## Performance & Scaling

### Redis Adapter for Horizontal Scaling

The Redis adapter allows multiple Socket.IO servers to share connected clients:

```text
┌─────────────┐        ┌─────────────┐
│  Client 1   │───────►│  Server A   │
└─────────────┘        └──────┬──────┘
                              │
┌─────────────┐               │      ┌─────────────┐
│  Client 2   │───────►│◄─────┴─────►│    Redis    │
└─────────────┘        │              └─────────────┘
                       │                     ▲
┌─────────────┐        │                     │
│  Client 3   │───────►│  Server B   │◄──────┘
└─────────────┘        └─────────────┘
```

**Setup**:

```javascript
import { createAdapter } from '@socket.io/redis-adapter';
import { createClient } from 'redis';

const pubClient = createClient({ url: 'redis://localhost:6379' });
const subClient = pubClient.duplicate();

await Promise.all([pubClient.connect(), subClient.connect()]);

io.adapter(createAdapter(pubClient, subClient));
```

**Benefits**:

- Events published on Server A reach clients on Server B
- Load balancing across multiple server instances
- Session persistence not required

### Connection Limits

Monitor active connections:

```javascript
io.on('connection', (socket) => {
  console.log(`Total connections: ${io.sockets.sockets.size}`);
});
```

Limit connections per user (optional):

```javascript
io.use(async (socket, next) => {
  const userId = socket.userId;
  
  const existingConnections = await io.in(`user:${userId}`).fetchSockets();
  
  if (existingConnections.length >= 3) {
    return next(new Error('Too many connections'));
  }
  
  next();
});
```

### Event Batching

For high-frequency events, batch updates:

```javascript
let pendingUpdates = [];

function batchPublish(eventName, data) {
  pendingUpdates.push({ eventName, data });
  
  // Flush every 100ms
  if (pendingUpdates.length === 1) {
    setTimeout(() => {
      const batch = pendingUpdates;
      pendingUpdates = [];
      
      io.emit('batch_update', batch);
    }, 100);
  }
}
```

---

## Summary

The real-time system provides:

✅ **Instant Updates** - Server-push notifications for long-running tasks  
✅ **User Isolation** - Room-based authorization ensures privacy  
✅ **Scalability** - Redis adapter enables horizontal scaling  
✅ **Reliability** - Automatic reconnection with exponential backoff  
✅ **Authentication** - JWT-based connection authentication  
✅ **Event Types** - Granular events for upload, analysis, completion  
✅ **React Integration** - Context-based Socket.IO provider  
✅ **Performance** - Efficient pub/sub with Redis  

**Next Steps**:

- [Services & Business Logic](./Services.md)
- [Middleware & Authentication](./Middleware.md)
- [API Routes Reference](./API-Routes.md)
