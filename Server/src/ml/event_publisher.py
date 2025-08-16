# src/ml/event_publisher.py

import os
import json
import redis
import logging

logger = logging.getLogger(__name__)

# Connect to Redis using the same environment variable as the Node.js backend
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
# The progress channel must match the one in the Node.js backend's constants.
VIDEO_PROGRESS_CHANNEL = "video-progress-events"

try:
    redis_client = redis.from_url(redis_url, decode_responses=True)
    # Ping to ensure connection is alive
    redis_client.ping()
    logger.info(f"✅ Redis client connected successfully for event publishing.")
except redis.exceptions.ConnectionError as e:
    logger.error(f"❌ Could not connect to Redis at {redis_url}. Event publishing will be disabled. Error: {e}")
    redis_client = None

def publish_progress(event_data: dict):
    """Publishes a progress event to the Redis channel if the client is available."""
    if redis_client:
        try:
            # The payload must match what the Node.js event service expects
            payload = json.dumps(event_data)
            redis_client.publish(VIDEO_PROGRESS_CHANNEL, payload)
            # logger.info(f"🚀 Published event '{event_data.get('event')}' to Redis channel '{VIDEO_PROGRESS_CHANNEL}'")
        except Exception as e:
            logger.error(f"Failed to publish progress event to Redis: {e}")