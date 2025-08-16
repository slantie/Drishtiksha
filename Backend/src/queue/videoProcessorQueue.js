// src/queue/videoProcessorQueue.js

import logger from "../utils/logger.js";
import { videoService } from "../services/video.service.js";

// ---
// NOTE: This is a simple, in-memory queue suitable for development and single-instance deployments.
// For production, consider replacing this with a robust message queue system like BullMQ with Redis
// to ensure job persistence, scalability, and retry capabilities.
// ---

const processingQueue = [];
let isProcessing = false;

/**
 * Adds a video analysis job to the processing queue.
 * @param {string} videoId - ID of the video to process.
 */
export const addVideoToQueue = (videoId) => {
    // Prevent adding duplicates
    if (processingQueue.some((job) => job.videoId === videoId)) {
        logger.warn(
            `Video ${videoId} is already in the processing queue. Skipping.`
        );
        return;
    }

    logger.info(`Adding video ${videoId} to the analysis queue.`);
    processingQueue.push({ videoId });

    // Start processing if not already running
    if (!isProcessing) {
        processQueue();
    }
};

const processQueue = async () => {
    if (isProcessing) return;
    isProcessing = true;
    logger.info("Starting to process the video analysis queue...");

    while (processingQueue.length > 0) {
        const job = processingQueue.shift();
        const { videoId } = job;

        logger.info(`Processing job for video ${videoId} from the queue.`);

        try {
            // The videoService will handle all analysis types and save the results.
            await videoService.runAllAnalysesForVideo(videoId);
            logger.info(`Successfully processed job for video ${videoId}.`);
        } catch (error) {
            logger.error(
                `Failed to process job for video ${videoId}: ${error.message}`
            );
            // In a real queue, you'd handle retries or move to a dead-letter queue.
            // For now, we'll mark the video as FAILED.
            await videoService.markVideoAsFailed(videoId, error.message);
        }
    }

    isProcessing = false;
    logger.info("Video processing queue is now empty.");
};

export const getQueueStatus = () => {
    return {
        isProcessing,
        queueLength: processingQueue.length,
        pendingJobs: processingQueue.map((job) => job.videoId),
    };
};
