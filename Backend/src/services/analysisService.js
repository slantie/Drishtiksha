/**
 * @fileoverview Service for handling the core video analysis logic.
 * This file contains the functions that simulate the deepfake detection process
 * and update the database accordingly. It's designed to be used by a background
 * worker or queue system.
 */

import { prisma } from "../config/database.js";
import logger from "../utils/logger.js";

// Models to be used for deepfake analysis
const ANALYSIS_MODELS = ["SIGLIPV1", "RPPG", "COLORCUES"];

/**
 * Generates mock analysis results for a video.
 * This is a simulated function that mimics the output of a real analysis model.
 * @param {string} filename - The name of the video file.
 * @param {number} fileSize - The size of the video file in bytes.
 * @param {string} model - The name of the analysis model used.
 * @returns {object} The mock analysis results.
 */
const generateMockAnalysis = (filename, fileSize, model) => {
    // Generate a random outcome to simulate real analysis
    const isLikelyReal = Math.random() > 0.3;
    const baseConfidence = isLikelyReal ? 0.75 + Math.random() * 0.2 : 0.6 + Math.random() * 0.3;
    const confidence = Math.min(0.98, Math.max(0.52, baseConfidence + (Math.random() - 0.5) * 0.1));
    
    // Simulate processing time based on file size
    const baseProcTime = 2.5 + (fileSize / (1024 * 1024)) * 0.1;
    const processingTime = baseProcTime + (Math.random() - 0.5) * 1.0;
    
    // Determine prediction based on confidence level
    const prediction = confidence > 0.75 ? "REAL" : "FAKE";

    return {
        prediction,
        confidence: parseFloat(confidence.toFixed(4)),
        processingTime: parseFloat(Math.max(1.2, processingTime).toFixed(2)),
        model: model,
        status: "COMPLETED",
    };
};

/**
 * Executes a full deepfake analysis for a given video with all available models.
 * @param {string} videoId - The ID of the video to analyze.
 * @param {string} userId - The ID of the user who owns the video.
 */
export const runVideoAnalysis = async (videoId, userId) => {
    logger.info(`Starting multi-model analysis for video ID: ${videoId}`);
    try {
        // Fetch the video details from the database
        const video = await prisma.video.findUnique({
            where: { id: videoId },
            include: { analyses: true }
        });

        if (!video) {
            logger.error(`Video with ID ${videoId} not found.`);
            return;
        }

        // Set the video status to PROCESSING
        await prisma.video.update({
            where: { id: videoId },
            data: { status: "PROCESSING" },
        });

        // Run analysis for each model
        for (const model of ANALYSIS_MODELS) {
            // Check if an analysis for this model already exists
            const existingAnalysis = video.analyses.find(a => a.model === model);
            if (existingAnalysis) {
                logger.info(`Analysis for model ${model} already exists for video ${videoId}. Skipping.`);
                continue;
            }

            logger.info(`Starting mock analysis for video ${videoId} with model ${model}.`);
            
            // Simulate a processing delay
            const processingDelay = 1000 + Math.random() * 2000;
            await new Promise((resolve) => setTimeout(resolve, processingDelay));

            // Generate mock results
            const mockResults = generateMockAnalysis(video.filename, video.size, model);

            // Create the new analysis record in the database
            await prisma.deepfakeAnalysis.create({
                data: {
                    videoId: videoId,
                    prediction: mockResults.prediction,
                    confidence: mockResults.confidence,
                    processingTime: mockResults.processingTime,
                    model: mockResults.model,
                    status: "COMPLETED",
                },
            });

            logger.info(`Analysis with model ${model} for video ${videoId} completed successfully.`);
        }

        // After all analyses are done, update the video status to ANALYZED
        await prisma.video.update({
            where: { id: videoId },
            data: { status: "ANALYZED" },
        });

        logger.info(`All analyses for video ${videoId} are now complete.`);

    } catch (error) {
        logger.error(`Failed to complete video analysis for ID ${videoId}: ${error.message}`);
        // If any error occurs, update the video status to FAILED
        await prisma.video.update({
            where: { id: videoId },
            data: { status: "FAILED" },
        });
    }
};
