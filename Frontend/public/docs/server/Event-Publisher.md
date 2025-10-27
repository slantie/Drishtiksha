# Event Publisher & Progress Tracking

## Overview

The **EventPublisher** (`src/ml/event_publisher.py`) is a singleton Redis-based pub/sub system that enables real-time progress reporting from ML inference operations to external consumers (typically a Node.js backend with WebSocket connections to frontend clients).

**Purpose:** Provide granular, type-safe progress updates during long-running deepfake detection tasks without blocking the inference thread.

**Design Pattern:** Singleton + Observer (Pub/Sub)

---

## Core Concepts

### Why Real-Time Progress Reporting?

Deepfake detection can take **seconds to minutes** depending on:

- Video length (e.g., 300 frames at ~0.1s/frame = 30 seconds)
- Model complexity (e.g., MFF-MoE with 7 experts)
- Hardware (CPU vs GPU, model optimization)

**Without progress tracking:**

- User sees spinning loader for 30+ seconds
- No indication of what's happening
- Poor UX, appears frozen

**With progress tracking:**

- Real-time updates: "Analyzed 50/300 frames"
- Phase indicators: "Generating visualization..."
- Estimated completion time
- Rich user experience

### Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    ML Server (Python)                        │
│                                                              │
│  ┌──────────────┐         ┌──────────────┐                 │
│  │   Detector   │ ──────> │EventPublisher│                 │
│  │  (analyze)   │  events │  (Singleton) │                 │
│  └──────────────┘         └──────┬───────┘                 │
│                                   │                          │
└───────────────────────────────────┼──────────────────────────┘
                                    │ Redis Pub/Sub
                                    │ (media:progress channel)
┌───────────────────────────────────┼──────────────────────────┐
│                    Backend (Node.js)                         │
│                                   │                          │
│  ┌────────────────────────────────▼───────────────┐         │
│  │        Redis Subscriber                        │         │
│  │  (listens to media:progress channel)           │         │
│  └────────────────┬───────────────────────────────┘         │
│                   │                                          │
│  ┌────────────────▼───────────────────────────────┐         │
│  │        WebSocket Server                        │         │
│  │  (broadcasts to connected clients)             │         │
│  └────────────────┬───────────────────────────────┘         │
└───────────────────┼──────────────────────────────────────────┘
                    │ WebSocket
┌───────────────────▼──────────────────────────────────────────┐
│                 Frontend (React)                             │
│                                                              │
│  ┌──────────────────────────────────────────┐               │
│  │  Progress Bar / Live Updates             │               │
│  │  "Analyzed 150/300 frames (50%)"         │               │
│  └──────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

---

## EventPublisher Class

### Singleton Implementation

```python
class EventPublisher:
    _instance = None
    _redis_client: Optional[redis.Redis] = None

    def __new__(cls):
        """
        Singleton pattern ensures only one EventPublisher instance exists.
        
        This prevents:
        - Multiple Redis connection pools
        - Duplicate event publications
        - Resource wastage
        """
        if cls._instance is None:
            cls._instance = super(EventPublisher, cls).__new__(cls)
            
            if settings.redis_url:
                logger.info(
                    f"Redis event publisher will use channel: "
                    f"'{settings.media_progress_channel_name}'"
                )
                cls._instance._connect()
            else:
                logger.warning(
                    "REDIS_URL is not set. Event publishing is disabled."
                )
        
        return cls._instance
```

**Singleton Benefits:**

- **Single Connection Pool**: All detectors share same Redis client
- **Consistent State**: No race conditions from multiple instances
- **Memory Efficiency**: One connection pool vs. N pools (N = number of models)
- **Configuration**: Settings loaded once at initialization

**Usage:**

```python
# Import the singleton instance
from src.ml.event_publisher import event_publisher

# All imports reference the SAME instance
event_publisher.publish(event)  # Detector 1
event_publisher.publish(event)  # Detector 2
# Both use the same Redis connection
```

### Connection Management

#### Initialization

```python
def _connect(self):
    """
    Initializes the Redis client with connection pooling and health checks.
    
    Configuration:
    - decode_responses=True: Automatic UTF-8 decoding (JSON-friendly)
    - health_check_interval=30: Ping every 30 seconds
    - socket_connect_timeout=5: Fail fast on connection issues
    - socket_keepalive=True: Prevent idle connection drops
    """
    logger.info(
        f"Attempting to connect to Redis for event publishing "
        f"at: {settings.redis_url}"
    )
    
    try:
        # Create connection pool (reusable connections)
        pool = redis.ConnectionPool.from_url(
            settings.redis_url,
            decode_responses=True,       # UTF-8 strings, not bytes
            health_check_interval=30,    # Ping every 30s
            socket_connect_timeout=5,    # Timeout after 5s
            socket_keepalive=True,       # Prevent idle disconnects
        )
        
        # Create Redis client from pool
        self._redis_client = redis.Redis(connection_pool=pool)
        
        # Verify connection
        self._redis_client.ping()
        
        logger.info(
            "✅ Redis client configured and connected successfully "
            "for event publishing."
        )
        
    except redis.exceptions.ConnectionError as e:
        logger.error(
            f"❌ Redis connection failed: {e}. "
            "Event publishing will be disabled until reconnect."
        )
        self._redis_client = None
        
    except Exception as e:
        logger.error(
            f"❌ An unexpected error occurred during Redis "
            f"client initialization: {e}",
            exc_info=True
        )
        self._redis_client = None
```

**Connection Pool Benefits:**

```python
# Without connection pool:
# Each publish() creates new TCP connection
publish(event1)  # Connect → Publish → Disconnect
publish(event2)  # Connect → Publish → Disconnect
publish(event3)  # Connect → Publish → Disconnect
# Total: 3 connections (expensive!)

# With connection pool:
# Reuses existing connections
publish(event1)  # Get from pool → Publish → Return to pool
publish(event2)  # Get from pool → Publish → Return to pool
publish(event3)  # Get from pool → Publish → Return to pool
# Total: 1 connection (efficient!)
```

#### Graceful Degradation

**If Redis is unavailable:**

```python
# EventPublisher initialization:
if settings.redis_url:
    cls._instance._connect()
else:
    logger.warning("REDIS_URL is not set. Event publishing is disabled.")
    # _redis_client remains None

# Publishing with no Redis:
def publish(self, event: ProgressEvent):
    if not self._redis_client:
        logger.debug("Skipping event publish: Redis client not connected.")
        return  # Silent no-op, doesn't crash
    
    # ... normal publishing logic
```

**Benefits:**

- **No Crashes**: ML server continues operating without Redis
- **Useful for Development**: Run server without Redis dependency
- **Resilient**: Temporary Redis outages don't stop inference

### Publishing Events

```python
def publish(self, event: ProgressEvent):
    """
    Publishes a validated progress event to the Redis channel.
    
    Args:
        event: Pydantic-validated ProgressEvent object
    
    Behavior:
    - Skips publish if Redis unavailable (graceful degradation)
    - Serializes event to JSON
    - Publishes to configured channel (default: "media:progress")
    - Auto-reconnects on connection errors
    - Logs all operations for debugging
    """
    # Guard: Skip if no Redis connection
    if not self._redis_client:
        logger.debug("Skipping event publish: Redis client not connected.")
        return

    try:
        # Serialize Pydantic model to JSON
        payload = event.model_dump_json()
        
        # Publish to Redis channel
        self._redis_client.publish(
            settings.media_progress_channel_name,  # "media:progress"
            payload
        )
        
        # Debug logging
        logger.debug(
            f"🚀 Published event '{event.event}' "
            f"(mediaId: {event.media_id}) to Redis."
        )
        
    except redis.exceptions.ConnectionError as e:
        # Connection lost mid-operation
        logger.error(
            f"❌ Could not publish to Redis, connection lost: {e}. "
            "Will attempt to reconnect on next publish."
        )
        self._redis_client = None  # Trigger reconnect on next publish
        
    except Exception as e:
        # Unexpected errors (JSON serialization, etc.)
        logger.error(
            f"❌ An unexpected error occurred while publishing "
            f"event '{event.event}' to Redis: {e}",
            exc_info=True
        )
```

**Error Handling Strategy:**

1. **Connection Errors** → Set `_redis_client = None` (triggers reconnect)
2. **Other Errors** → Log with traceback, continue execution
3. **No Redis** → Silent no-op (graceful degradation)

---

## Event Schemas

### ProgressEvent

**Primary schema for all progress updates:**

```python
class ProgressEvent(BaseModel):
    """
    Main schema for all progress events published to Redis.
    Enforces consistent structure across all event types.
    """
    media_id: str = Field(
        ...,
        description="Unique identifier for the media being processed (e.g., videoId)"
    )
    
    user_id: Optional[str] = Field(
        None,
        description="Identifier for the user who initiated the request"
    )
    
    event: EventType = Field(
        ...,
        description="Specific event type that occurred"
    )
    
    message: str = Field(
        ...,
        description="Human-readable message describing the event"
    )
    
    data: EventData
```

**Field Descriptions:**

| Field | Type | Required | Purpose |
|-------|------|----------|---------|
| `media_id` | `str` | ✅ | Unique ID for the media file (e.g., video UUID) |
| `user_id` | `str` | ❌ | User who initiated analysis (for multi-tenancy) |
| `event` | `EventType` | ✅ | Type-safe event category (see EventType enum) |
| `message` | `str` | ✅ | Human-readable status message |
| `data` | `EventData` | ✅ | Flexible payload with progress details |

### EventType

**Literal type ensuring valid event categories:**

```python
EventType = Literal[
    "FRAME_ANALYSIS_PROGRESS",         # Video frame processing updates
    "AUDIO_EXTRACTION_START",          # Audio extraction beginning
    "AUDIO_EXTRACTION_COMPLETE",       # Audio extraction finished
    "SPECTROGRAM_GENERATION_START",    # Spectrogram creation beginning
    "SPECTROGRAM_GENERATION_COMPLETE", # Spectrogram creation finished
    "ANALYSIS_COMPLETE",               # Full analysis finished (success)
    "ANALYSIS_FAILED",                 # Analysis failed (error)
]
```

**Benefits of Literal Types:**

```python
# ✅ Valid (type-checked at runtime)
event_publisher.publish(ProgressEvent(
    event="FRAME_ANALYSIS_PROGRESS",
    ...
))

# ❌ Invalid (Pydantic raises ValidationError)
event_publisher.publish(ProgressEvent(
    event="INVALID_EVENT_NAME",  # Not in EventType enum
    ...
))
```

### EventData

**Flexible payload schema:**

```python
class EventData(BaseModel):
    """
    Flexible data payload for events.
    Contains model-specific information and progress metrics.
    """
    model_name: str  # Required: Which model is processing
    
    progress: Optional[int] = None   # Current progress (e.g., 50 frames)
    total: Optional[int] = None      # Total items (e.g., 300 frames)
    
    details: Optional[Dict[str, Any]] = None  # Additional metadata
```

**Field Usage:**

| Field | Purpose | Example |
|-------|---------|---------|
| `model_name` | Identify which model is processing | `"SIGLIP-LSTM-V4"` |
| `progress` | Current completion count | `150` (frames analyzed) |
| `total` | Total expected count | `300` (total frames) |
| `details` | Arbitrary metadata | `{"phase": "frame_processing", "fps": 30}` |

---

## Usage Patterns

### Basic Progress Updates

**Video frame analysis:**

```python
from src.ml.event_publisher import event_publisher
from src.ml.schemas import ProgressEvent, EventData

def analyze(self, media_path: str, **kwargs):
    video_id = kwargs.get("video_id")
    user_id = kwargs.get("user_id")
    
    total_frames = 300
    
    for i in range(total_frames):
        # Process frame...
        
        # Publish progress every 10 frames
        if (i + 1) % 10 == 0 and video_id and user_id:
            event_publisher.publish(ProgressEvent(
                media_id=video_id,
                user_id=user_id,
                event="FRAME_ANALYSIS_PROGRESS",
                message=f"Analyzed {i + 1}/{total_frames} frames",
                data=EventData(
                    model_name=self.config.model_name,
                    progress=i + 1,
                    total=total_frames
                )
            ))
```

**Frontend receives:**

```json
{
  "media_id": "abc123-video-uuid",
  "user_id": "user456",
  "event": "FRAME_ANALYSIS_PROGRESS",
  "message": "Analyzed 150/300 frames",
  "data": {
    "model_name": "SIGLIP-LSTM-V4",
    "progress": 150,
    "total": 300,
    "details": null
  }
}
```

### Multi-Phase Progress

**Audio analysis with multiple stages:**

```python
def analyze(self, media_path: str, **kwargs):
    video_id = kwargs.get("video_id")
    user_id = kwargs.get("user_id")
    
    # Phase 1: Audio Extraction
    if video_id and user_id:
        event_publisher.publish(ProgressEvent(
            media_id=video_id,
            user_id=user_id,
            event="AUDIO_EXTRACTION_START",
            message="Extracting audio track from video",
            data=EventData(
                model_name=self.config.model_name,
                details={"phase": "audio_extraction"}
            )
        ))
    
    audio = extract_audio(media_path)
    
    if video_id and user_id:
        event_publisher.publish(ProgressEvent(
            media_id=video_id,
            user_id=user_id,
            event="AUDIO_EXTRACTION_COMPLETE",
            message="Audio extraction complete",
            data=EventData(
                model_name=self.config.model_name,
                details={
                    "phase": "audio_extraction_complete",
                    "duration_seconds": audio.duration
                }
            )
        ))
    
    # Phase 2: Spectrogram Generation
    if video_id and user_id:
        event_publisher.publish(ProgressEvent(
            media_id=video_id,
            user_id=user_id,
            event="SPECTROGRAM_GENERATION_START",
            message="Generating mel-spectrogram",
            data=EventData(
                model_name=self.config.model_name,
                details={"phase": "spectrogram_generation"}
            )
        ))
    
    spectrogram = generate_spectrogram(audio)
    
    if video_id and user_id:
        event_publisher.publish(ProgressEvent(
            media_id=video_id,
            user_id=user_id,
            event="SPECTROGRAM_GENERATION_COMPLETE",
            message="Spectrogram generation complete",
            data=EventData(
                model_name=self.config.model_name,
                details={
                    "phase": "spectrogram_complete",
                    "chunks_processed": len(spectrogram)
                }
            )
        ))
    
    # Phase 3: Analysis
    result = self._analyze_spectrogram(spectrogram, video_id, user_id)
    
    # Phase 4: Completion
    if video_id and user_id:
        event_publisher.publish(ProgressEvent(
            media_id=video_id,
            user_id=user_id,
            event="ANALYSIS_COMPLETE",
            message=f"Analysis complete: {result.prediction} ({result.confidence:.2%})",
            data=EventData(
                model_name=self.config.model_name,
                details={
                    "prediction": result.prediction,
                    "confidence": result.confidence,
                    "processing_time": result.processing_time
                }
            )
        ))
    
    return result
```

**Timeline of events:**

```text
1. AUDIO_EXTRACTION_START
2. AUDIO_EXTRACTION_COMPLETE (2.5s later)
3. SPECTROGRAM_GENERATION_START
4. SPECTROGRAM_GENERATION_COMPLETE (1.2s later)
5. FRAME_ANALYSIS_PROGRESS (chunk 1/10)
6. FRAME_ANALYSIS_PROGRESS (chunk 2/10)
   ... (every chunk)
10. FRAME_ANALYSIS_PROGRESS (chunk 10/10)
11. ANALYSIS_COMPLETE
```

### Error Handling

**Publishing failure events:**

```python
def analyze(self, media_path: str, **kwargs):
    video_id = kwargs.get("video_id")
    user_id = kwargs.get("user_id")
    
    try:
        # ... analysis logic
        
        result = self._run_inference(media_path)
        
        # Success event
        if video_id and user_id:
            event_publisher.publish(ProgressEvent(
                media_id=video_id,
                user_id=user_id,
                event="ANALYSIS_COMPLETE",
                message=f"Analysis complete: {result.prediction}",
                data=EventData(
                    model_name=self.config.model_name,
                    details={"prediction": result.prediction}
                )
            ))
        
        return result
        
    except Exception as e:
        # Failure event
        if video_id and user_id:
            event_publisher.publish(ProgressEvent(
                media_id=video_id,
                user_id=user_id,
                event="ANALYSIS_FAILED",
                message=f"Analysis failed: {str(e)}",
                data=EventData(
                    model_name=self.config.model_name,
                    details={
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )
            ))
        
        raise  # Re-raise for API error handling
```

### Conditional Publishing

**Only publish if tracking IDs provided:**

```python
# Pattern used throughout codebase:
if video_id and user_id:
    event_publisher.publish(...)

# Why?
# - CLI tools don't provide video_id/user_id
# - Only web API requests need progress tracking
# - Prevents unnecessary Redis traffic
```

**Example:**

```python
# API request (has tracking IDs):
POST /analyze
{
  "media": <file>,
  "mediaId": "abc123",     # ✅ Provided
  "userId": "user456"       # ✅ Provided
}
→ Events published

# CLI usage (no tracking IDs):
python -m src.cli.predict video.mp4 --model SIGLIP-LSTM-V4
→ video_id=None, user_id=None
→ No events published (silent)
```

---

## Integration with Backend

### Node.js Redis Subscriber

**Backend listens to same channel:**

```javascript
// backend/src/services/redis.js

const redis = require('redis');

const subscriber = redis.createClient({
  url: process.env.REDIS_URL
});

subscriber.connect();

subscriber.subscribe('media:progress', (message) => {
  const event = JSON.parse(message);
  
  // Broadcast to WebSocket clients
  broadcastToUser(event.user_id, {
    type: 'ANALYSIS_PROGRESS',
    mediaId: event.media_id,
    event: event.event,
    message: event.message,
    data: event.data
  });
});
```

### WebSocket Broadcasting

**Backend sends to connected clients:**

```javascript
// backend/src/websocket/server.js

function broadcastToUser(userId, data) {
  const userSockets = connectedClients.get(userId);
  
  if (userSockets) {
    userSockets.forEach(socket => {
      socket.emit('progress-update', data);
    });
  }
}
```

### Frontend Consumption

**React component receives updates:**

```jsx
// frontend/src/hooks/useAnalysisProgress.js

import { useEffect, useState } from 'react';
import { socket } from '../lib/socket';

export function useAnalysisProgress(mediaId) {
  const [progress, setProgress] = useState({
    current: 0,
    total: 0,
    message: '',
    phase: ''
  });

  useEffect(() => {
    const handleProgress = (data) => {
      if (data.mediaId === mediaId) {
        setProgress({
          current: data.data.progress || 0,
          total: data.data.total || 100,
          message: data.message,
          phase: data.event
        });
      }
    };

    socket.on('progress-update', handleProgress);

    return () => {
      socket.off('progress-update', handleProgress);
    };
  }, [mediaId]);

  return progress;
}

// Usage in component:
function AnalysisView({ mediaId }) {
  const progress = useAnalysisProgress(mediaId);

  return (
    <div>
      <ProgressBar 
        value={progress.current} 
        max={progress.total} 
      />
      <p>{progress.message}</p>
      <small>Phase: {progress.phase}</small>
    </div>
  );
}
```

---

## Configuration

### Environment Variables

```bash
# .env file

REDIS_URL=redis://localhost:6379
MEDIA_PROGRESS_CHANNEL_NAME=media:progress
```

**Settings Object:**

```python
# src/config.py

class Settings(BaseSettings):
    redis_url: Optional[str] = Field(
        None,
        env="REDIS_URL",
        description="Redis connection URL for event publishing"
    )
    
    media_progress_channel_name: str = Field(
        "media:progress",
        env="MEDIA_PROGRESS_CHANNEL_NAME",
        description="Redis Pub/Sub channel name for progress events"
    )
```

**Docker Compose:**

```yaml
# docker-compose.yml

services:
  app:
    environment:
      - REDIS_URL=redis://redis:6379
      - MEDIA_PROGRESS_CHANNEL_NAME=media:progress
    depends_on:
      - redis
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

---

## Best Practices

### 1. Publish Frequency

**Balance between UX and performance:**

```python
# ❌ Too frequent (every frame, 300 events for 300 frames)
for i in range(300):
    event_publisher.publish(...)  # 300 Redis publishes!

# ✅ Optimal (every 5-10 frames, ~30-60 events)
for i in range(300):
    if (i + 1) % 10 == 0:
        event_publisher.publish(...)  # 30 Redis publishes
```

**Guidelines:**

- **Video frames**: Every 5-10 frames (60-120 FPS → ~5-10 updates/sec)
- **Audio chunks**: Every chunk (usually 5-10 chunks total)
- **Phases**: Every major phase transition (extraction, generation, analysis)

### 2. Detailed Phase Information

**Use `details` for debugging:**

```python
event_publisher.publish(ProgressEvent(
    media_id=video_id,
    user_id=user_id,
    event="FRAME_ANALYSIS_PROGRESS",
    message=f"Analyzed {i + 1}/{total} frames",
    data=EventData(
        model_name=self.config.model_name,
        progress=i + 1,
        total=total,
        details={
            "phase": "frame_processing",
            "current_frame_faces": len(faces),
            "total_faces_so_far": total_faces_detected,
            "avg_score": np.mean(scores),
            "fps": total / elapsed_time
        }
    )
))
```

**Benefits:**

- **Debugging**: Rich context for troubleshooting
- **Monitoring**: Track performance metrics
- **UX**: Show detailed progress (e.g., "150 faces detected")

### 3. Conditional Publishing

**Only publish when needed:**

```python
def analyze(self, media_path: str, **kwargs):
    video_id = kwargs.get("video_id")
    user_id = kwargs.get("user_id")
    
    # Extract once, check everywhere
    should_publish = video_id and user_id
    
    if should_publish:
        event_publisher.publish(...)
    
    # ... analysis logic
    
    if should_publish:
        event_publisher.publish(...)
```

### 4. Consistent Naming

**Use model_name from config:**

```python
# ❌ Hardcoded class name
data=EventData(
    model_name="SiglipLSTMV4",  # Wrong! Internal class name
    ...
)

# ✅ User-friendly name from config
data=EventData(
    model_name=self.config.model_name,  # ✅ "SIGLIP-LSTM-V4"
    ...
)
```

---

## Debugging

### Logging Levels

```python
# Debug logging (verbose):
logger.debug(
    f"🚀 Published event '{event.event}' "
    f"(mediaId: {event.media_id}) to Redis."
)

# Info logging (important events):
logger.info(
    f"Redis event publisher will use channel: "
    f"'{settings.media_progress_channel_name}'"
)

# Error logging (failures):
logger.error(
    f"❌ Could not publish to Redis, connection lost: {e}. "
    "Will attempt to reconnect on next publish."
)
```

### Testing Event Publishing

**Manual Redis monitoring:**

```bash
# Terminal 1: Start Redis CLI
redis-cli

# Subscribe to channel
SUBSCRIBE media:progress

# Terminal 2: Run ML server and trigger analysis
curl -X POST http://localhost:8000/analyze \
  -H "X-API-Key: your_key" \
  -F "media=@video.mp4" \
  -F "model=SIGLIP-LSTM-V4"

# Terminal 1: See events in real-time
1) "message"
2) "media:progress"
3) "{\"media_id\":\"abc123\",\"event\":\"FRAME_ANALYSIS_PROGRESS\",...}"
```

### Verifying Event Structure

```python
# Use Pydantic validation to catch errors early

try:
    event = ProgressEvent(
        media_id="test123",
        user_id="user456",
        event="INVALID_EVENT",  # ❌ Not in EventType enum
        message="Test",
        data=EventData(model_name="TEST")
    )
except ValidationError as e:
    print(e)
    # ValidationError: 1 validation error for ProgressEvent
    # event
    #   Input should be 'FRAME_ANALYSIS_PROGRESS', ...
```

---

## Summary

The EventPublisher provides a robust, type-safe real-time progress reporting system:

✅ **Singleton Pattern** - Single Redis connection shared across all models  
✅ **Graceful Degradation** - Works without Redis (silent no-op)  
✅ **Type Safety** - Pydantic validation ensures correct event structure  
✅ **Connection Pooling** - Efficient Redis connection reuse  
✅ **Auto-Reconnect** - Handles Redis outages gracefully  
✅ **Flexible Payloads** - EventData supports arbitrary metadata  
✅ **Multi-Phase Support** - Track complex workflows (extraction → analysis → visualization)  
✅ **Conditional Publishing** - Only publish when tracking IDs provided  

**Key Integration Points:**

1. **ML Server** → Redis Pub/Sub → **Node.js Backend** → WebSocket → **React Frontend**
2. Enables rich UX with real-time progress bars, phase indicators, and ETA calculations
3. Decoupled architecture allows ML server to run independently of backend

**Usage Pattern:**

```python
from src.ml.event_publisher import event_publisher

if video_id and user_id:
    event_publisher.publish(ProgressEvent(
        media_id=video_id,
        user_id=user_id,
        event="FRAME_ANALYSIS_PROGRESS",
        message="Analyzed 150/300 frames",
        data=EventData(
            model_name=self.config.model_name,
            progress=150,
            total=300
        )
    ))
```
