// src/config/queue.js

import { Queue, FlowProducer } from "bullmq";
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

export const videoQueue = new Queue(VIDEO_PROCESSING_QUEUE_NAME, {
    connection: redisConnection,
    defaultJobOptions: {
        attempts: 3,
        backoff: { type: "exponential", delay: 5000 },
    },
});

export const videoFlowProducer = new FlowProducer({
    connection: redisConnection,
});

videoQueue.on("error", (err) => {
    logger.error(`BullMQ Queue Error: ${err.message}`);
});

export const addVideoToQueue = async (videoId) => {
    await videoQueue.add("analysis-flow", { videoId }, { jobId: videoId });
    logger.info(`Video ${videoId} 'analysis-flow' job added to the queue.`);
};

export const getQueueStatus = async () => {
    return {
        pendingJobs: await videoQueue.getWaitingCount(),
        activeJobs: await videoQueue.getActiveCount(),
        completedJobs: await videoQueue.getCompletedCount(),
        failedJobs: await videoQueue.getFailedCount(),
        delayedJobs: await videoQueue.getDelayedCount(),
    };
};
