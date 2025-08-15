// src/queue/videoProcessorQueue.js

import logger from "../utils/logger.js";
import { videoService } from "../services/video.service.js";

const processingQueue = [];
let isProcessing = false;

/**
 * Adds a video analysis job to the processing queue
 * @param {Object} jobData - Job data containing videoId, userId, and optional analysis configuration
 * @param {string} jobData.videoId - ID of the video to process
 * @param {string} jobData.userId - ID of the user who uploaded the video
 * @param {Object} jobData.analysisConfig - Optional analysis configuration
 * @param {Array} jobData.analysisConfig.types - Analysis types to run (default: ["QUICK"])
 * @param {Array} jobData.analysisConfig.models - Models to use (default: all available)
 * @param {boolean} jobData.analysisConfig.enableVisualization - Whether to generate visualizations
 */
export const videoProcessorQueue = (jobData) => {
    logger.info(`Adding video ${jobData.videoId} to the processing queue.`);

    // Set default analysis configuration if not provided
    if (!jobData.analysisConfig) {
        jobData.analysisConfig = {
            types: ["QUICK"], // Default to quick analysis only
            models: undefined, // Use all available models
            enableVisualization: false, // Don't generate visualizations by default
        };
    }

    processingQueue.push(jobData);

    if (!isProcessing) {
        processQueue();
    }
};

/**
 * Processes the video analysis queue
 */
const processQueue = async () => {
    isProcessing = true;

    while (processingQueue.length > 0) {
        const job = processingQueue.shift();
        const { videoId, userId, analysisConfig } = job;

        logger.info(
            `Processing job for video ${videoId} from the queue with config:`,
            analysisConfig
        );

        try {
            // Call the enhanced analysis method with configuration
            await videoService.runFullAnalysis(videoId, analysisConfig);
            logger.info(`Successfully processed job for video ${videoId}.`);
        } catch (error) {
            logger.error(
                `Failed to process job for video ${videoId}: ${error.message}`
            );
            // Optionally add retry logic here in the future
        }
    }

    isProcessing = false;
    logger.info("Video processing queue is now empty.");
};

/**
 * Gets the current queue status
 * @returns {Object} Queue status information
 */
export const getQueueStatus = () => {
    return {
        queueLength: processingQueue.length,
        isProcessing,
        pendingJobs: processingQueue.map((job) => ({
            videoId: job.videoId,
            userId: job.userId,
            analysisTypes: job.analysisConfig?.types || ["QUICK"],
        })),
    };
};

/**
 * Adds a high-priority analysis job to the front of the queue
 * @param {Object} jobData - Job data (same format as videoProcessorQueue)
 */
export const addPriorityJob = (jobData) => {
    logger.info(
        `Adding priority video ${jobData.videoId} to the front of the processing queue.`
    );

    // Set default analysis configuration if not provided
    if (!jobData.analysisConfig) {
        jobData.analysisConfig = {
            types: ["QUICK"],
            models: undefined,
            enableVisualization: false,
        };
    }

    processingQueue.unshift(jobData); // Add to front of queue

    if (!isProcessing) {
        processQueue();
    }
};
