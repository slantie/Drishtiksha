/**
 * @fileoverview A simple in-memory queue for video processing tasks.
 * This file is a placeholder to resolve the 'ERR_MODULE_NOT_FOUND' error.
 * For a production environment, you would use a dedicated queue system like BullMQ or RabbitMQ.
 */

import logger from "../utils/logger.js";
import { runVideoAnalysis } from "../services/analysisService.js";

// A simple in-memory array to act as a task queue
const processingQueue = [];
let isProcessing = false;

/**
 * Adds a new video processing job to the queue.
 * @param {object} videoData - The video data to be processed.
 * @param {string} videoData.videoId - The ID of the video to analyze.
 * @param {string} videoData.userId - The ID of the user who uploaded the video.
 */
export const videoProcessorQueue = (videoData) => {
  logger.info(`Adding video ${videoData.videoId} to the processing queue.`);
  processingQueue.push(videoData);
  // Start the processing loop if it's not already running
  if (!isProcessing) {
    processQueue();
  }
};

/**
 * Processes the next video in the queue.
 */
const processQueue = async () => {
  isProcessing = true;
  while (processingQueue.length > 0) {
    const job = processingQueue.shift();
    logger.info(`Processing video ${job.videoId} from the queue.`);
    try {
      await runVideoAnalysis(job.videoId, job.userId);
      logger.info(`Successfully processed video ${job.videoId}.`);
    } catch (error) {
      logger.error(`Failed to process video ${job.videoId}: ${error.message}`);
      // In a real application, you might handle retries here
    }
  }
  isProcessing = false;
  logger.info("Video processing queue is now empty.");
};