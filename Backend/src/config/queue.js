// src/config/queue.js

import { Queue, FlowProducer } from "bullmq";
import { VIDEO_PROCESSING_QUEUE_NAME } from "./constants.js";
import logger from "../utils/logger.js";

const createRedisConnection = () => {
    const redisUrl = process.env.REDIS_URL;

    if (redisUrl) {
        const redisUri = new URL(redisUrl);
        const connectionOptions = {
            host: redisUri.hostname,
            port: parseInt(redisUri.port, 10),
            password: redisUri.password,
        };

        if (redisUri.protocol === "rediss:") {
            connectionOptions.tls = {
                rejectUnauthorized: false,
            };
            logger.info(
                `BullMQ is configured for a secure (TLS) Redis connection to ${redisUri.hostname}.`
            );
        } else {
            logger.info(
                `BullMQ is configured for a standard Redis connection to ${redisUri.hostname}.`
            );
        }
        return connectionOptions;
    }

    logger.warn(`REDIS_URL not found. BullMQ is connecting to local Redis.`);
    return {
        host: "localhost",
        port: 6379,
    };
};

export const redisConnection = createRedisConnection();

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

export const addAnalysisFlowToQueue = async (videoId, childJobs) => {
    await videoFlowProducer.add({
        name: "finalize-analysis",
        queueName: VIDEO_PROCESSING_QUEUE_NAME,
        data: { videoId },
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