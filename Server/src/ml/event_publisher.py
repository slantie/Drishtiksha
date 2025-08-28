# src/ml/event_publisher.py

import os
import json
import redis
import logging

# Use __name__ for a module-specific logger, standard best practice.
logger = logging.getLogger(__name__)

# Get Redis URL from environment variables
redis_url = os.getenv("REDIS_URL")

MEDIA_PROGRESS_CHANNEL = "media-progress-events"
redis_client = None

# This block attempts to initialize the Redis client on module import.
# It is designed to fail gracefully if Redis is unavailable, allowing the main app to start.
if not redis_url:
    logger.warning("⚠️ REDIS_URL environment variable is not set. Event publishing will be disabled.")
else:
    logger.info(f"Attempting to connect to Redis for event publishing at: {redis_url}")
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
        response = redis_client.ping()
        
        if response:
            logger.info("✅ Redis client configured and connected successfully for event publishing.")
        else:
            # This case indicates a problem even if the connection object was created.
            logger.warning("⚠️ Connected to Redis, but the initial PING command failed. Event publishing may be unreliable.")

    except redis.exceptions.AuthenticationError:
        logger.error("❌ Redis CONNECTION FAILED: Authentication error. Check REDIS_URL password.")
        redis_client = None
    except redis.exceptions.ConnectionError as e:
        logger.error(f"❌ Redis CONNECTION FAILED: A connection error occurred. Event publishing will be disabled. Error: {e}")
        redis_client = None
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred during Redis client initialization: {e}")
        redis_client = None


def publish_progress(event_data: dict):
    """
    Publishes a progress event to the Redis channel if the client is available.
    Fails silently if Redis is not connected, but logs the error.
    """
    if not redis_client:
        # Log a debug message if publish is attempted when Redis is not available
        logger.debug(f"Skipping event publish: Redis client not initialized or connected. Event: {event_data.get('event')}")
        return

    try:
        payload = json.dumps(event_data)
        redis_client.publish(MEDIA_PROGRESS_CHANNEL, payload)
        # Log successful publishes at debug level to avoid excessive log noise in production
        logger.debug(f"🚀 Published event '{event_data.get('event')}' (mediaId: {event_data.get('videoId') or event_data.get('mediaId')}) to Redis channel '{MEDIA_PROGRESS_CHANNEL}'.")

    except redis.exceptions.ConnectionError as e:
        # This will catch errors if the connection is lost despite the health checks
        logger.error(f"❌ Could not publish to Redis, connection lost: {e}. Event: {event_data.get('event')}. Will attempt to reconnect on next publish if configured.")
        # Optionally set redis_client = None here if you want to completely disable until manual restart/reconnect logic
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred while publishing event '{event_data.get('event')}' to Redis: {e}", exc_info=True)