// src/services/event.service.js

import logger from '../utils/logger.js';
import { config } from '../config/env.js';
import { redisSubscriber } from '../config/redis.js';

export function initializeRedisListener(io) {
    redisSubscriber.subscribe(config.MEDIA_PROGRESS_CHANNEL_NAME, (err) => {
        if (err) {
            logger.error(`[EventService] Failed to subscribe to Redis channel '${config.MEDIA_PROGRESS_CHANNEL_NAME}':`, err);
        } else {
            logger.info(`[EventService] Subscribed to '${config.MEDIA_PROGRESS_CHANNEL_NAME}' for real-time progress.`);
        }
    });

    redisSubscriber.on('message', (channel, message) => {
        if (channel === config.MEDIA_PROGRESS_CHANNEL_NAME) {
            try {
                const progressData = JSON.parse(message);
                if (progressData.user_id) {
                    io.to(progressData.user_id).emit('progress_update', progressData);
                }
            } catch (error) {
                logger.error('[EventService] Failed to parse progress event message:', error);
            }
        }
    });
}