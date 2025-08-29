# src/ml/event_publisher.py

import os
import redis
import logging
from typing import Optional

from src.ml.schemas import ProgressEvent

logger = logging.getLogger(__name__)

# --- Configuration ---
REDIS_URL = os.getenv("REDIS_URL")
# REFACTOR: Provide a default channel name to prevent crashes if the .env variable is not set.
MEDIA_PROGRESS_CHANNEL = os.getenv("MEDIA_PROGRESS_CHANNEL_NAME", "media-progress-events")

class EventPublisher:
    _instance = None
    _redis_client: Optional[redis.Redis] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EventPublisher, cls).__new__(cls)
            if REDIS_URL:
                # Log the channel name being used for clarity.
                logger.info(f"Redis event publisher will use channel: '{MEDIA_PROGRESS_CHANNEL}'")
                cls._instance._connect()
            else:
                logger.warning("REDIS_URL is not set. Event publishing is disabled.")
        return cls._instance

    # ... (rest of the file is unchanged) ...
    def _connect(self):
        """Initializes the Redis client and connection pool."""
        logger.info(f"Attempting to connect to Redis for event publishing at: {REDIS_URL}")
        try:
            pool = redis.ConnectionPool.from_url(
                REDIS_URL,
                decode_responses=True,
                health_check_interval=30,
                socket_connect_timeout=5,
                socket_keepalive=True,
            )
            self._redis_client = redis.Redis(connection_pool=pool)
            self._redis_client.ping()
            logger.info("✅ Redis client configured and connected successfully for event publishing.")
        except redis.exceptions.ConnectionError as e:
            logger.error(f"❌ Redis connection failed: {e}. Event publishing will be disabled until reconnect.")
            self._redis_client = None
        except Exception as e:
            logger.error(f"❌ An unexpected error occurred during Redis client initialization: {e}", exc_info=True)
            self._redis_client = None

    def publish(self, event: ProgressEvent):
        """Publishes a validated progress event to the Redis channel."""
        if not self._redis_client:
            logger.debug("Skipping event publish: Redis client not connected.")
            return

        try:
            payload = event.model_dump_json()
            self._redis_client.publish(MEDIA_PROGRESS_CHANNEL, payload)
            logger.debug(f"🚀 Published event '{event.event}' (mediaId: {event.media_id}) to Redis.")
        except redis.exceptions.ConnectionError as e:
            logger.error(f"❌ Could not publish to Redis, connection lost: {e}. Will attempt to reconnect on next publish.")
            self._redis_client = None
        except Exception as e:
            logger.error(f"❌ An unexpected error occurred while publishing event '{event.event}' to Redis: {e}", exc_info=True)

# Create a single, globally accessible instance.
event_publisher = EventPublisher()