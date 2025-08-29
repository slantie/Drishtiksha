// src/services/event.service.js

import Redis from "ioredis";
import logger from "../utils/logger.js";
import { redisConnection } from "../config/queue.js";

const publisher = new Redis({ ...redisConnection, lazyConnect: true });
const subscriber = new Redis({ ...redisConnection, lazyConnect: true });

/**
 * Emits a progress event to the Redis Pub/Sub channel.
 * @param {object} payload - The progress data to send.
 */
const emitProgress = async (payload) => {
    try {
        if (publisher.status !== "ready") await publisher.connect();

        // UPDATED: The log message now refers to the generic mediaId.
        logger.info(
            `[EventBus] Publishing event '${payload.event}' for media ${payload.mediaId}`
        );
        publisher.publish(process.env.MEDIA_PROGRESS_CHANNEL_NAME || "media-progress-channel", JSON.stringify(payload));
    } catch (error) {
        logger.error(`[EventBus] Failed to publish event: ${error.message}`);
    }
};

/**
 * Listens for progress events on the Redis Pub/Sub channel.
 * This function is called by the main API server to receive and forward events to clients.
 * @param {function} callback - The function to execute when a message is received.
 */
const listenForProgress = async (callback) => {
    try {
        if (subscriber.status !== "ready") await subscriber.connect();

        subscriber.subscribe(process.env.MEDIA_PROGRESS_CHANNEL_NAME || "media-progress-channel", (err) => {
            if (err) {
                logger.error("Failed to subscribe to progress channel:", err);
            } else {
                logger.info(
                    `[EventBus] Subscribed to '${process.env.MEDIA_PROGRESS_CHANNEL_NAME || "media-progress-channel"}' for real-time progress updates.`
                );
            }
        });

        subscriber.on("message", (channel, message) => {
            if (channel === process.env.MEDIA_PROGRESS_CHANNEL_NAME || "media-progress-channel") {
                try {
                    const progressData = JSON.parse(message);
                    callback(progressData);
                } catch (error) {
                    logger.error(
                        "Failed to parse progress event message:",
                        error
                    );
                }
            }
        });

        subscriber.on("error", (err) => {
            logger.error(
                `[EventBus] Subscriber connection error: ${err.message}`
            );
        });
    } catch (error) {
        logger.error(
            `[EventBus] Failed to listen for progress: ${error.message}`
        );
    }
};

export const eventService = {
    emitProgress,
    listenForProgress,
};
