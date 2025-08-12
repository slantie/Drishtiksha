// src/queue/videoProcessorQueue.js

import logger from "../utils/logger.js";
import { videoService } from "../services/video.service.js";

const processingQueue = [];
let isProcessing = false;

export const videoProcessorQueue = (jobData) => {
    logger.info(`Adding video ${jobData.videoId} to the processing queue.`);
    processingQueue.push(jobData);
    if (!isProcessing) {
        processQueue();
    }
};

const processQueue = async () => {
    isProcessing = true;
    while (processingQueue.length > 0) {
        const job = processingQueue.shift();
        logger.info(`Processing job for video ${job.videoId} from the queue.`);
        try {
            // Call the method on the new video service
            await videoService.runFullAnalysis(job.videoId);
            logger.info(`Successfully processed job for video ${job.videoId}.`);
        } catch (error) {
            logger.error(
                `Failed to process job for video ${job.videoId}: ${error.message}`
            );
        }
    }
    isProcessing = false;
    logger.info("Video processing queue is now empty.");
};
