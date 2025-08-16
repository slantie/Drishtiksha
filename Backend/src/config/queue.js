// src/config/queue.js

import { Queue } from "bullmq";
import { VIDEO_PROCESSING_QUEUE_NAME } from "./constants.js";
import logger from "../utils/logger.js";

const redisConnection = {
    host: process.env.REDIS_URL
        ? new URL(process.env.REDIS_URL).hostname
        : "localhost",
    port: process.env.REDIS_URL
        ? parseInt(new URL(process.env.REDIS_URL).port)
        : 6379,
};

// This is the single queue instance for the application.
export const videoQueue = new Queue(VIDEO_PROCESSING_QUEUE_NAME, {
    connection: redisConnection,
    defaultJobOptions: {
        attempts: 3,
        backoff: { type: "exponential", delay: 5000 },
    },
});

videoQueue.on("error", (err) => {
    logger.error(`BullMQ Queue Error: ${err.message}`);
});

// ADDED: This function adds a job to the queue.
// REASON: Consolidates queue interaction logic into this file.
export const addVideoToQueue = async (videoId) => {
    await videoQueue.add("process-video", { videoId }, { jobId: videoId });
    logger.info(`Video ${videoId} job added to the processing queue.`);
};

// ADDED: This function retrieves queue status for monitoring.
// REASON: Consolidates queue interaction logic into this file.
export const getQueueStatus = async () => {
    return {
        pendingJobs: await videoQueue.getWaitingCount(),
        activeJobs: await videoQueue.getActiveCount(),
        completedJobs: await videoQueue.getCompletedCount(),
        failedJobs: await videoQueue.getFailedCount(),
        delayedJobs: await videoQueue.getDelayedCount(),
    };
};
