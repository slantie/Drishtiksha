// src/services/event.service.js

import logger from "../utils/logger.js";
import { config } from "../config/env.js";
import { redisSubscriber } from "../config/redis.js";

export function initializeRedisListener(io) {
  redisSubscriber.subscribe(config.MEDIA_PROGRESS_CHANNEL_NAME, (err) => {
    if (err) {
      logger.error(
        `[EventService] Failed to subscribe to Redis channel '${config.MEDIA_PROGRESS_CHANNEL_NAME}':`,
        err
      );
    } else {
      logger.info(
        `[EventService] ✅ Subscribed to '${config.MEDIA_PROGRESS_CHANNEL_NAME}' for real-time progress updates.`
      );
    }
  });

  redisSubscriber.on("message", (channel, message) => {
    if (channel === config.MEDIA_PROGRESS_CHANNEL_NAME) {
      try {
        const progressData = JSON.parse(message);

        // Validate the event structure
        if (!progressData.media_id) {
          logger.warn(
            "[EventService] Received progress event without media_id:",
            progressData
          );
          return;
        }

        if (!progressData.user_id) {
          logger.warn(
            "[EventService] Received progress event without user_id:",
            progressData
          );
          return;
        }

        // Log the event for debugging (can be disabled in production)
        logger.debug(
          `[EventService] 📢 Broadcasting '${progressData.event}' event for media_id: ${progressData.media_id}, user_id: ${progressData.user_id}`
        );

        // Convert Server event format to Frontend-compatible format
        const frontendEvent = {
          mediaId: progressData.media_id, // Convert snake_case to camelCase
          userId: progressData.user_id,
          event: progressData.event,
          message: progressData.message,
          data: progressData.data,
          timestamp: new Date().toISOString(),
        };

        // Emit to the specific user's room
        io.to(progressData.user_id).emit("progress_update", frontendEvent);

        // Optional: Log successful broadcast
        logger.debug(
          `[EventService] ✅ Successfully broadcasted event to user ${progressData.user_id}`
        );
      } catch (error) {
        logger.error(
          "[EventService] ❌ Failed to parse or broadcast progress event message:",
          error
        );
        logger.error("[EventService] Raw message:", message);
      }
    }
  });

  // Handle Redis connection errors
  redisSubscriber.on("error", (err) => {
    logger.error("[EventService] ❌ Redis subscriber connection error:", err);
  });

  redisSubscriber.on("ready", () => {
    logger.info("[EventService] ✅ Redis subscriber connection ready");
  });

  redisSubscriber.on("close", () => {
    logger.warn("[EventService] ⚠️ Redis subscriber connection closed");
  });
}
