// src/services/event.service.js

import Redis from "ioredis";
import logger from "../utils/logger.js";
import { VIDEO_PROGRESS_CHANNEL } from "../config/constants.js";

// REASON: It's a best practice for Redis Pub/Sub to use two separate clients:
// one for publishing and one for subscribing, to avoid race conditions and connection blocking.
const publisher = new Redis(process.env.REDIS_URL, { lazyConnect: true });
const subscriber = new Redis(process.env.REDIS_URL, { lazyConnect: true });

/**
 * Emits a progress event to the Redis Pub/Sub channel.
 * This function is called by the worker to report its progress.
 * @param {object} payload - The progress data to send.
 * @param {string} payload.videoId - The ID of the video being processed.
 * @param {string} payload.userId - The ID of the user who owns the video.
 * @param {string} payload.event - The name of the event (e.g., 'ANALYSIS_STARTED').
 * @param {string} payload.message - A user-friendly message.
 * @param {object} [payload.data] - Optional additional data for the frontend.
 */
const emitProgress = async (payload) => {
    if (publisher.status !== "ready") await publisher.connect();
    logger.info(
        `[EventBus] Publishing event '${payload.event}' for video ${payload.videoId}`
    );
    publisher.publish(VIDEO_PROGRESS_CHANNEL, JSON.stringify(payload));
};

/**
 * Listens for progress events on the Redis Pub/Sub channel.
 * This function is called by the main API server to receive and forward events to clients.
 * @param {function} callback - The function to execute when a message is received.
 */
const listenForProgress = async (callback) => {
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
};

export const eventService = {
    emitProgress,
    listenForProgress,
};
