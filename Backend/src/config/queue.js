// src/config/queue.js (Corrected Version)

import { Queue, FlowProducer } from "bullmq";
import { VIDEO_PROCESSING_QUEUE_NAME } from "./constants.js";
import logger from "../utils/logger.js";

const redisUrl = process.env.REDIS_URL;
let redisConnection;

// Check if a REDIS_URL is provided in the environment
if (redisUrl) {
    const redisUri = new URL(redisUrl);

    // Construct the connection object correctly for a secure Upstash connection
    redisConnection = {
        host: redisUri.hostname,
        port: parseInt(redisUri.port, 10),
        password: redisUri.password,
        // This is the crucial part for enabling TLS/SSL encryption
        tls: {
            rejectUnauthorized: false, // Necessary for many cloud providers
        },
    };
    logger.info(
        `BullMQ is configured to connect to Redis at ${redisUri.hostname}`
    );
} else {
    // Fallback for local development without a REDIS_URL
    redisConnection = {
        host: "localhost",
        port: 6379,
    };
    logger.warn(`REDIS_URL not found. BullMQ is connecting to local Redis.`);
}

// Now, create your queue and producer instances with the correct connection object
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

// The rest of your functions remain the same
export const addVideoToQueue = async (videoId) => {
    await videoQueue.add("analysis-flow", { videoId }, { jobId: videoId });
    logger.info(`Video ${videoId} 'analysis-flow' job added to the queue.`);
};

export const addAnalysisFlowToQueue = async (videoId, childJobs) => {
    await videoFlowProducer.add({
        name: "finalize-analysis",
        queueName: VIDEO_PROCESSING_QUEUE_NAME,
        data: { videoId, totalAnalysesAttempted: childJobs.length },
        opts: { jobId: `${videoId}-finalizer` },
        children: childJobs,
    });
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
