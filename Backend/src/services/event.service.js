// src/services/event.service.js

import Redis from "ioredis";
import logger from "../utils/logger.js";
import { VIDEO_PROGRESS_CHANNEL } from "../config/constants.js";
// --- MODIFIED: Import the single, centralized Redis connection object ---
import { redisConnection } from "../config/queue.js";

// --- REMOVED: All old, duplicate connection logic is now gone. ---

// REASON: It's a best practice for Pub/Sub to use two separate clients.
// --- MODIFIED: Both clients now use the exact same connection configuration. ---
const publisher = new Redis({ ...redisConnection, lazyConnect: true });
const subscriber = new Redis({ ...redisConnection, lazyConnect: true });

/**
 * Emits a progress event to the Redis Pub/Sub channel.
 * This function is called by the worker to report its progress.
 * @param {object} payload - The progress data to send.
 */
const emitProgress = async (payload) => {
    try {
        if (publisher.status !== "ready") await publisher.connect();
        logger.info(
            `[EventBus] Publishing event '${payload.event}' for video ${payload.videoId}`
        );
        publisher.publish(VIDEO_PROGRESS_CHANNEL, JSON.stringify(payload));
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

        subscriber.subscribe(VIDEO_PROGRESS_CHANNEL, (err) => {
            if (err) {
                logger.error("Failed to subscribe to progress channel:", err);
            } else {
                logger.info(
                    `[EventBus] Subscribed to '${VIDEO_PROGRESS_CHANNEL}' for real-time progress updates.`
                );
            }
        });

        subscriber.on("message", (channel, message) => {
            if (channel === VIDEO_PROGRESS_CHANNEL) {
                try {
                    const progressData = JSON.parse(message);
                    callback(progressData);
                } catch (error) {
                    logger.error("Failed to parse progress event message:", error);
                }
            }
        });

        // Handle connection errors on the subscriber
        subscriber.on('error', (err) => {
            logger.error(`[EventBus] Subscriber connection error: ${err.message}`);
        });

    } catch (error) {
        logger.error(`[EventBus] Failed to listen for progress: ${error.message}`);
    }
};

export const eventService = {
    emitProgress,
    listenForProgress,
};