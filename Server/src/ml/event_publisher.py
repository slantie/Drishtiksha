# src/ml/event_publisher.py (Improved Version)

import os
import json
import redis
import logging

logger = logging.getLogger(__name__)

# Get Redis URL from environment variables
redis_url = os.getenv("REDIS_URL")

if "redis:" in  redis_url:
    logger.warning("Detected 'redis:' in REDIS_URL, converting to 'rediss:' for secure connection.")
    redis_url = redis_url.replace("redis:", "rediss:")

VIDEO_PROGRESS_CHANNEL = "video-progress-events"
redis_client = None

# Only initialize the client if REDIS_URL is set
if redis_url:
    try:
        # Use a connection pool with health checks to keep the connection alive
        # and handle network drops gracefully.
        redis_client = redis.from_url(
            redis_url,
            decode_responses=True,
            health_check_interval=30,  # Sends a PING every 30s to prevent timeouts
            socket_connect_timeout=5,    # Timeout for establishing a connection
            socket_keepalive=True,       # Enable TCP keepalives
            retry_on_timeout=True        # Retry on timeout errors
        )
        
        # Initial ping to confirm the credentials and connectivity at startup
        redis_client.ping()
        logger.info("✅ Redis client configured and connected successfully for event publishing.")

    except redis.exceptions.ConnectionError as e:
        logger.error(f"❌ Initial connection to Redis at {redis_url} failed. Event publishing will be disabled. Error: {e}")
        redis_client = None
else:
    logger.warning("⚠️ REDIS_URL not set. Event publishing is disabled.")


def publish_progress(event_data: dict):
    """Publishes a progress event to the Redis channel if the client is available."""
    if not redis_client:
        # Silently fail if Redis was never configured or failed to connect initially
        return

    try:
        payload = json.dumps(event_data)
        redis_client.publish(VIDEO_PROGRESS_CHANNEL, payload)
        # Uncomment for debugging if needed
        # logger.info(f"🚀 Published event '{event_data.get('event')}' to Redis channel '{VIDEO_PROGRESS_CHANNEL}'")
    
    except redis.exceptions.ConnectionError as e:
        # This will catch errors if the connection is lost despite the health checks
        logger.error(f"Could not publish to Redis, connection lost: {e}. Will attempt to reconnect on next publish.")
    
    except Exception as e:
        logger.error(f"An unexpected error occurred while publishing to Redis: {e}")