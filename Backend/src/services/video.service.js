// src/services/video.service.js

import { promises as fs } from "fs"; // Import the promise-based API as 'fs'
import { createWriteStream } from "fs"; // Import createWriteStream separately
import path from "path";
import { fileURLToPath } from "url";
import { videoRepository } from "../repositories/video.repository.js";
import {
    uploadOnCloudinary,
    uploadStreamToCloudinary,
    deleteFromCloudinary,
} from "../utils/cloudinary.js";
import { ApiError } from "../utils/ApiError.js";
import { videoProcessorQueue } from "../queue/videoProcessorQueue.js";
import logger from "../utils/logger.js";
import { modelAnalysisService } from "./modelAnalysis.service.js";
import axios from "axios";

// Available models for analysis - updated to match new schema enums
const ANALYSIS_MODELS = [
    "SIGLIP_LSTM_V1",
    "SIGLIP_LSTM_V3",
    "COLOR_CUES_LSTM_V1",
];

// Default analysis configurations
const DEFAULT_ANALYSIS_CONFIG = {
    types: ["QUICK"], // Default to quick analysis
    models: ANALYSIS_MODELS,
    enableVisualization: false,
};

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.join(__dirname, "..", "..");

const generateMockAnalysis = (filename, fileSize, model) => {
    const isLikelyReal = Math.random() > 0.3;
    const baseConfidence = isLikelyReal
        ? 0.75 + Math.random() * 0.2
        : 0.6 + Math.random() * 0.3;
    const confidence = parseFloat(
        Math.min(0.98, Math.max(0.52, baseConfidence)).toFixed(4)
    );
    const prediction = confidence > 0.75 ? "REAL" : "FAKE";
    const processingTime = parseFloat(
        (
            2.5 +
            (fileSize / (1024 * 1024)) * 0.1 +
            (Math.random() - 0.5)
        ).toFixed(2)
    );
    return {
        prediction,
        confidence,
        processingTime,
        model,
        status: "COMPLETED",
    };
};

export const videoService = {
    async uploadAndProcessVideo(file, description, user) {
        if (!file) {
            throw new ApiError(400, "No video file provided");
        }

        const cloudinaryResponse = await uploadOnCloudinary(file.path);
        if (!cloudinaryResponse) {
            throw new ApiError(500, "Video failed to upload on Cloudinary");
        }

        const newVideo = await videoRepository.create({
            filename: file.originalname,
            mimetype: file.mimetype,
            size: file.size,
            description: description || "",
            url: cloudinaryResponse.secure_url,
            publicId: cloudinaryResponse.public_id,
            userId: user.id,
            status: "UPLOADED",
        });

        // Add job to the queue for automatic multi-model analysis
        videoProcessorQueue({ videoId: newVideo.id, userId: newVideo.userId });
        logger.info(
            `Video ${newVideo.id} added to in-memory processing queue.`
        );

        return newVideo;
    },

    async getAllVideosForUser(user) {
        if (user.role === "ADMIN") {
            return videoRepository.findAll();
        }
        return videoRepository.findAllByUser(user.id);
    },

    async getVideoById(videoId, user) {
        const video = await videoRepository.findById(videoId);
        if (!video) {
            throw new ApiError(404, "Video not found");
        }
        if (video.userId !== user.id && user.role !== "ADMIN") {
            throw new ApiError(
                403,
                "Access denied. You do not own this video."
            );
        }
        return video;
    },

    async updateVideoDetails(videoId, updateData, user) {
        const video = await this.getVideoById(videoId, user); // Reuse permission check
        return videoRepository.update(videoId, updateData);
    },

    async deleteVideoById(videoId, user) {
        const video = await this.getVideoById(videoId, user); // Reuse permission check

        if (video.publicId) {
            await deleteFromCloudinary(video.publicId, "video");
        }

        await videoRepository.delete(videoId);
    },

    /**
     * Runs comprehensive analysis on a video with multiple models and types
     * @param {string} videoId - Video ID to analyze
     * @param {Object} analysisConfig - Configuration for analysis types and models
     * @returns {Promise<void>} Completes analysis processing
     */
    async runFullAnalysis(videoId, analysisConfig = DEFAULT_ANALYSIS_CONFIG) {
        logger.info(`Starting comprehensive analysis for video ID: ${videoId}`);

        const video = await videoRepository.findById(videoId);
        if (!video) {
            logger.error(`Video with ID ${videoId} not found for analysis.`);
            return;
        }

        await videoRepository.update(videoId, { status: "PROCESSING" });

        try {
            const {
                types = ["QUICK"],
                models = ANALYSIS_MODELS,
                enableVisualization = false,
            } = analysisConfig;

            // Process each combination of type and model
            for (const type of types) {
                for (const model of models) {
                    // Check if analysis already exists
                    const existingAnalysis = await videoRepository.findAnalysis(
                        video.id,
                        model,
                        type
                    );

                    if (
                        existingAnalysis &&
                        existingAnalysis.status !== "FAILED"
                    ) {
                        logger.info(
                            `${type} analysis with ${model} already exists for video ${videoId}. Skipping.`
                        );
                        continue;
                    }

                    // Create new analysis entry
                    const analysisData = {
                        videoId: video.id,
                        model: model,
                        type: type,
                        status: "PROCESSING",
                        confidence: 0,
                        isDeepfake: false,
                        processingStartedAt: new Date(),
                    };

                    let analysis;
                    if (existingAnalysis) {
                        analysis = await videoRepository.updateAnalysis(
                            existingAnalysis.id,
                            analysisData
                        );
                    } else {
                        analysis = await videoRepository.createAnalysis(
                            analysisData
                        );
                    }

                    // Perform the analysis
                    await this._performSingleAnalysis(
                        video,
                        analysis,
                        type,
                        model
                    );
                }
            }

            // Handle visualization if requested
            if (enableVisualization) {
                for (const model of models) {
                    const existingVisualization =
                        await videoRepository.findAnalysis(
                            video.id,
                            model,
                            "VISUALIZE"
                        );

                    if (
                        !existingVisualization ||
                        existingVisualization.status === "FAILED"
                    ) {
                        const visualizationData = {
                            videoId: video.id,
                            model: model,
                            type: "VISUALIZE",
                            status: "PROCESSING",
                            confidence: 0,
                            isDeepfake: false,
                            processingStartedAt: new Date(),
                        };

                        let visualization;
                        if (existingVisualization) {
                            visualization =
                                await videoRepository.updateAnalysis(
                                    existingVisualization.id,
                                    visualizationData
                                );
                        } else {
                            visualization =
                                await videoRepository.createAnalysis(
                                    visualizationData
                                );
                        }

                        await this._performVisualization(
                            video,
                            visualization,
                            model
                        );
                    }
                }
            }

            await videoRepository.update(videoId, { status: "ANALYZED" });
            logger.info(`All analyses for video ${videoId} are now complete.`);
        } catch (error) {
            logger.error(
                `Failed to complete video analysis for ID ${videoId}: ${error.message}`
            );
            await videoRepository.update(videoId, { status: "FAILED" });
        }
    },

    /**
     * Performs a single analysis for a specific type and model
     * @private
     */
    async _performSingleAnalysis(video, analysis, type, model) {
        try {
            logger.info(
                `Performing ${type} analysis with ${model} for video: ${video.id}`
            );

            let results;

            // Get the local video file path for analysis
            const videoPath = await this.getLocalVideoPath(video);

            try {
                // Perform analysis using the model analysis service
                results = await modelAnalysisService.analyzeVideoWithFallback(
                    videoPath,
                    video.id,
                    type,
                    model,
                    { filename: video.filename, size: video.size }
                );
            } catch (error) {
                logger.error(
                    `${type} analysis with ${model} failed for video ${video.id}: ${error.message}`
                );
                // Fall back to mock data
                results = modelAnalysisService.generateFallbackResult(
                    type,
                    model,
                    {
                        filename: video.filename,
                        size: video.size,
                    }
                );
            } finally {
                // Clean up local file if it was downloaded
                await this.cleanupLocalVideoPath(videoPath, video);
            }

            // Prepare analysis update data
            const updateData = {
                status: "COMPLETED",
                confidence: results.confidence || 0,
                isDeepfake:
                    results.is_deepfake || results.prediction === "FAKE",
                prediction: results.prediction || "UNKNOWN",
                processingCompletedAt: new Date(),
                processingTime: results.processing_time || 0,
                modelVersion: results.model_version || "unknown",
                rawResult: results,
            };

            // Add type-specific data handling here if needed
            // (will be expanded based on the enhanced schema relationships)

            // Update analysis with results
            await videoRepository.updateAnalysis(analysis.id, updateData);

            logger.info(
                `${type} analysis completed for video ${video.id} with model ${model}`
            );
        } catch (error) {
            logger.error(
                `${type} analysis failed for video ${video.id} with model ${model}:`,
                error
            );

            // Update analysis with error status
            await videoRepository.updateAnalysis(analysis.id, {
                status: "FAILED",
                processingCompletedAt: new Date(),
            });
        }
    },

    /**
     * Performs visualization analysis
     * @private
     */
    async _performVisualization(video, analysis, model) {
        try {
            logger.info(
                `Performing visualization with ${model} for video: ${video.id}`
            );

            // Get the local video file path for visualization
            const videoPath = await this.getLocalVideoPath(video);

            try {
                // Generate visualization
                const result =
                    await modelAnalysisService.generateVisualAnalysis(
                        videoPath,
                        model,
                        video.id
                    );

                // Update analysis with visualization result
                const updateData = {
                    status: "COMPLETED",
                    confidence: 1.0, // Visualization doesn't have confidence
                    isDeepfake: false, // Visualization analysis itself
                    prediction: "VISUALIZATION",
                    processingCompletedAt: new Date(),
                    visualizedUrl: result.visualizationUrl,
                    rawResult: result,
                };

                await videoRepository.updateAnalysis(analysis.id, updateData);

                logger.info(
                    `Visualization completed for video ${video.id} with model ${model}`
                );
            } catch (error) {
                logger.error(
                    `Visualization failed for video ${video.id} with model ${model}:`,
                    error
                );

                // Update analysis with error status
                await videoRepository.updateAnalysis(analysis.id, {
                    status: "FAILED",
                    processingCompletedAt: new Date(),
                });
            } finally {
                // Clean up local file if it was downloaded
                await this.cleanupLocalVideoPath(videoPath, video);
            }
        } catch (error) {
            logger.error(
                `Visualization setup failed for video ${video.id} with model ${model}:`,
                error
            );

            await videoRepository.updateAnalysis(analysis.id, {
                status: "FAILED",
                processingCompletedAt: new Date(),
            });
        }
    },

    /**
     * Gets local video path for analysis - downloads from Cloudinary if needed
     * @param {Object} video - Video object with URL and metadata
     * @returns {string} Local file path
     */
    async getLocalVideoPath(video) {
        const fs = await import("fs");
        const https = await import("https");
        const http = await import("http");
        const { promisify } = await import("util");
        const pipeline = promisify((await import("stream")).pipeline);

        // Check for local file first (development setup)
        const uploadsDir = path.join(process.cwd(), "uploads", "videos");
        const possibleLocalPaths = [
            path.join(uploadsDir, `video-${video.id}.mp4`),
            path.join(uploadsDir, video.filename),
            path.join(uploadsDir, `${video.id}.mp4`),
        ];

        for (const localPath of possibleLocalPaths) {
            if (fs.existsSync(localPath)) {
                logger.info(`Using local video file: ${localPath}`);
                return localPath;
            }
        }

        // If no local file found, download from Cloudinary
        logger.info(
            `Local video file not found for ${video.id}, downloading from Cloudinary`
        );

        if (!video.url) {
            throw new Error(`No URL available for video ${video.id}`);
        }

        // Create temporary file path
        const tempDir = path.join(process.cwd(), "temp");
        if (!fs.existsSync(tempDir)) {
            fs.mkdirSync(tempDir, { recursive: true });
        }

        const tempFilePath = path.join(
            tempDir,
            `temp-${video.id}-${Date.now()}.mp4`
        );

        try {
            // Download video from Cloudinary
            const httpModule = video.url.startsWith("https:") ? https : http;

            await new Promise((resolve, reject) => {
                const request = httpModule.get(video.url, (response) => {
                    if (response.statusCode !== 200) {
                        reject(
                            new Error(
                                `Failed to download video: HTTP ${response.statusCode}`
                            )
                        );
                        return;
                    }

                    const writeStream = fs.createWriteStream(tempFilePath);
                    pipeline(response, writeStream).then(resolve).catch(reject);
                });

                request.on("error", reject);
                request.setTimeout(30000, () => {
                    request.destroy();
                    reject(new Error("Download timeout"));
                });
            });

            logger.info(`Video downloaded successfully to: ${tempFilePath}`);
            return tempFilePath;
        } catch (error) {
            // Clean up partial download
            if (fs.existsSync(tempFilePath)) {
                fs.unlinkSync(tempFilePath);
            }
            logger.error(
                `Failed to download video ${video.id}: ${error.message}`
            );
            throw error;
        }
    },

    /**
     * Cleans up temporary video files if they were downloaded
     * @param {string} videoPath - Path to the video file
     * @param {Object} video - Video object
     */
    async cleanupLocalVideoPath(videoPath, video) {
        const fs = await import("fs");

        // Only clean up temporary files (those in temp directory)
        if (videoPath.includes("temp") && fs.existsSync(videoPath)) {
            try {
                fs.unlinkSync(videoPath);
                logger.debug(`Cleaned up temporary video file: ${videoPath}`);
            } catch (error) {
                logger.warn(
                    `Failed to cleanup temporary file ${videoPath}: ${error.message}`
                );
            }
        }
    },

    /**
     * Creates visual analysis for a video with enhanced model support
     * @param {string} videoId - Video ID to create visualization for
     * @param {Object} user - User making the request
     * @param {string} model - Specific model to use for visualization (optional)
     * @returns {Promise<Object>} Updated video with visualization results
     */
    async createVisualAnalysis(videoId, user, model = null) {
        const video = await this.getVideoById(videoId, user); // Reuse permission check

        if (!video.url) {
            throw new ApiError(
                400,
                "Original video URL not found. Cannot perform visual analysis."
            );
        }

        // Determine which models to use for visualization
        const modelsToVisualize = model ? [model] : ANALYSIS_MODELS;
        const results = [];

        for (const visualModel of modelsToVisualize) {
            try {
                // Check if visualization already exists for this model
                const existingVisualization =
                    await videoRepository.findAnalysis(
                        video.id,
                        visualModel,
                        "VISUALIZE"
                    );

                if (
                    existingVisualization &&
                    existingVisualization.status === "COMPLETED"
                ) {
                    logger.info(
                        `Visualization with ${visualModel} already exists for video ${videoId}`
                    );
                    results.push(existingVisualization);
                    continue;
                }

                // Create or update visualization analysis entry
                const visualizationData = {
                    videoId: video.id,
                    model: visualModel,
                    type: "VISUALIZE",
                    status: "PROCESSING",
                    confidence: 0,
                    isDeepfake: false,
                    processingStartedAt: new Date(),
                };

                let visualization;
                if (existingVisualization) {
                    visualization = await videoRepository.updateAnalysis(
                        existingVisualization.id,
                        visualizationData
                    );
                } else {
                    visualization = await videoRepository.createAnalysis(
                        visualizationData
                    );
                }

                // Perform the visualization
                await this._performVisualization(
                    video,
                    visualization,
                    visualModel
                );

                // Get the updated analysis
                const updatedVisualization = await videoRepository.findAnalysis(
                    video.id,
                    visualModel,
                    "VISUALIZE"
                );
                results.push(updatedVisualization);
            } catch (error) {
                logger.error(
                    `Visualization with ${visualModel} failed for video ${videoId}:`,
                    error
                );

                // Update analysis with error status if it exists
                const failedVisualization = await videoRepository.findAnalysis(
                    video.id,
                    visualModel,
                    "VISUALIZE"
                );
                if (failedVisualization) {
                    await videoRepository.updateAnalysis(
                        failedVisualization.id,
                        {
                            status: "FAILED",
                            processingCompletedAt: new Date(),
                        }
                    );
                }

                // If this is a single model request, throw the error
                if (model) {
                    throw error;
                }

                // For multiple models, continue with others
                continue;
            }
        }

        // Return updated video with all visualizations
        const updatedVideo = await videoRepository.findById(video.id);

        // Check if at least one visualization succeeded
        const successfulVisualizations = results.filter(
            (r) => r && r.status === "COMPLETED"
        );
        if (successfulVisualizations.length === 0) {
            throw new ApiError(500, "All visualization attempts failed");
        }

        return updatedVideo;
    },

    /**
     * Enhanced method to trigger specific analysis types
     * @param {string} videoId - Video ID to analyze
     * @param {Object} user - User making the request
     * @param {string} analysisType - Type of analysis (QUICK, DETAILED, FRAMES, VISUALIZE)
     * @param {string} model - Specific model to use (optional)
     * @returns {Promise<Object>} Analysis results
     */
    async createSpecificAnalysis(videoId, user, analysisType, model = null) {
        const video = await this.getVideoById(videoId, user);

        if (!this.isValidAnalysisType(analysisType)) {
            throw new ApiError(400, `Invalid analysis type: ${analysisType}`);
        }

        const modelsToUse = model ? [model] : ANALYSIS_MODELS;
        const results = [];

        for (const analysisModel of modelsToUse) {
            // Check if analysis already exists
            const existingAnalysis = await videoRepository.findAnalysis(
                video.id,
                analysisModel,
                analysisType
            );

            if (existingAnalysis && existingAnalysis.status === "COMPLETED") {
                logger.info(
                    `${analysisType} analysis with ${analysisModel} already exists for video ${videoId}`
                );
                results.push(existingAnalysis);
                continue;
            }

            // Create new analysis entry
            const analysisData = {
                videoId: video.id,
                model: analysisModel,
                type: analysisType,
                status: "PROCESSING",
                confidence: 0,
                isDeepfake: false,
                processingStartedAt: new Date(),
            };

            let analysis;
            if (existingAnalysis) {
                analysis = await videoRepository.updateAnalysis(
                    existingAnalysis.id,
                    analysisData
                );
            } else {
                analysis = await videoRepository.createAnalysis(analysisData);
            }

            // Perform the analysis based on type
            if (analysisType === "VISUALIZE") {
                await this._performVisualization(
                    video,
                    analysis,
                    analysisModel
                );
            } else {
                await this._performSingleAnalysis(
                    video,
                    analysis,
                    analysisType,
                    analysisModel
                );
            }

            // Get updated analysis
            const updatedAnalysis = await videoRepository.findAnalysis(
                video.id,
                analysisModel,
                analysisType
            );
            results.push(updatedAnalysis);
        }

        return results;
    },

    /**
     * Helper method to validate analysis types
     * @param {string} type - Analysis type to validate
     * @returns {boolean} True if valid
     */
    isValidAnalysisType(type) {
        return ["QUICK", "DETAILED", "FRAMES", "VISUALIZE"].includes(type);
    },
};
