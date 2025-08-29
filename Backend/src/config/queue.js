// src/config/queue.js

import { Queue, FlowProducer } from "bullmq";
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

// RENAMED: from videoQueue to mediaQueue
export const mediaQueue = new Queue(
  process.env.MEDIA_PROCESSING_QUEUE_NAME || "media-processing-queue",
  {
    connection: redisConnection,
    defaultJobOptions: {
      attempts: 3,
      backoff: { type: "exponential", delay: 5000 },
    },
  }
);

// RENAMED: from videoFlowProducer to mediaFlowProducer
export const mediaFlowProducer = new FlowProducer({
  connection: redisConnection,
});

mediaQueue.on("error", (err) => {
  logger.error(`BullMQ Queue Error: ${err.message}`);
});

// REMOVED: addVideoToQueue is obsolete as flows are now created directly in the service.

/**
 * Adds a BullMQ flow to the queue.
 * A flow consists of a parent job (the finalizer) that only runs after all
 * its child jobs (the individual model analyses) have completed.
 *
 * @param {string} mediaId - The ID of the media item being processed.
 * @param {Array<object>} childJobs - An array of child job definitions for each analysis.
 */
// RENAMED: from addAnalysisFlowToQueue to addMediaAnalysisFlow
export const addMediaAnalysisFlow = async (mediaId, childJobs) => {
  await mediaFlowProducer.add({
    name: "finalize-analysis",
    queueName:
      process.env.MEDIA_PROCESSING_QUEUE_NAME || "media-processing-queue",
    // UPDATED: Passing mediaId and the total number of jobs to the finalizer.
    data: { mediaId: mediaId, totalAnalysesAttempted: childJobs.length },
    opts: { jobId: `${mediaId}-finalizer` },
    children: childJobs,
  });
  logger.info(
    `Media ${mediaId} analysis flow with ${childJobs.length} jobs added to the queue.`
  );
};

export const getQueueStatus = async () => {
  // This function remains the same, but we'll use the renamed mediaQueue variable.
  return {
    pendingJobs: await mediaQueue.getWaitingCount(),
    activeJobs: await mediaQueue.getActiveCount(),
    completedJobs: await mediaQueue.getCompletedCount(),
    failedJobs: await mediaQueue.getFailedCount(),
    delayedJobs: await mediaQueue.getDelayedCount(),
  };
};
