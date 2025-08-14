// src/services/modelAnalysis.service.js

import axios from "axios";
import FormData from "form-data";
import fs from "fs";
import path from "path";
import { config } from "dotenv";
import { ApiError } from "../utils/ApiError.js";
import logger from "../utils/logger.js";

// Load environment variables
config();

class ModelAnalysisService {
    constructor() {
        this.serverUrl = process.env.SERVER_URL || "http://localhost:8000";
        this.apiKey = process.env.SERVER_API_KEY;

        logger.info(`Model service initialized with URL: ${this.serverUrl}`);

        if (!this.apiKey) {
            logger.warn(
                "SERVER_API_KEY not found in environment variables. LSTM model analysis will be disabled."
            );
        } else {
            logger.info("SERVER_API_KEY found - LSTM model analysis enabled");
        }
    }

    /**
     * Checks if the model analysis service is available
     * @returns {boolean} True if service is configured and available
     */
    isAvailable() {
        return Boolean(this.apiKey && this.serverUrl);
    }

    /**
     * Checks the health of the model server
     * @returns {Promise<Object>} Health status from the server
     */
    async checkHealth() {
        try {
            const response = await axios.get(`${this.serverUrl}/health`, {
                timeout: 5000,
            });
            return response.data;
        } catch (error) {
            logger.error(`Model server health check failed: ${error.message}`);
            throw new ApiError(503, "Model analysis service is unavailable");
        }
    }

    /**
     * Gets model information from the server
     * @returns {Promise<Object>} Model information
     */
    async getModelInfo() {
        try {
            const response = await axios.get(`${this.serverUrl}/model/info`, {
                timeout: 5000,
            });
            return response.data;
        } catch (error) {
            logger.error(`Failed to get model info: ${error.message}`);
            throw new ApiError(503, "Unable to retrieve model information");
        }
    }

    /**
     * Analyzes a video using the LSTM model
     * @param {string} videoPath - Path to the video file
     * @param {string} videoId - Unique identifier for the video
     * @returns {Promise<Object>} Analysis results from the LSTM model
     */
    async analyzeVideo(videoPath, videoId) {
        if (!this.isAvailable()) {
            throw new ApiError(503, "Model analysis service is not configured");
        }

        if (!fs.existsSync(videoPath)) {
            throw new ApiError(400, "Video file not found");
        }

        try {
            logger.info(`Starting LSTM model analysis for video: ${videoId}`);
            const startTime = Date.now();

            // Create form data for the request
            const formData = new FormData();
            formData.append("video", fs.createReadStream(videoPath));
            formData.append("video_id", videoId);

            // Make request to the model server
            const response = await axios.post(
                `${this.serverUrl}/analyze`,
                formData,
                {
                    headers: {
                        ...formData.getHeaders(),
                        "X-API-Key": this.apiKey,
                    },
                    timeout: 120000, // 2 minutes timeout for model processing
                    maxContentLength: Infinity,
                    maxBodyLength: Infinity,
                }
            );

            const processingTime = (Date.now() - startTime) / 1000;
            logger.info(
                `LSTM model analysis completed for video ${videoId} in ${processingTime}s`
            );

            // Transform the response to match our expected format
            const result = response.data;
            if (!result.success) {
                throw new ApiError(500, "Model analysis failed");
            }

            return {
                prediction: result.result.prediction,
                confidence: result.result.confidence,
                processingTime: result.result.processing_time,
                model: "LSTM_SIGLIP",
                modelVersion: result.result.model_version,
                status: "COMPLETED",
                timestamp: new Date().toISOString(),
            };
        } catch (error) {
            logger.error(
                `LSTM model analysis failed for video ${videoId}: ${error.message}`
            );

            if (error.response) {
                // Server responded with error status
                const status = error.response.status;
                const message =
                    error.response.data?.detail ||
                    error.response.data?.message ||
                    "Model analysis failed";
                throw new ApiError(status, message);
            } else if (error.code === "ECONNREFUSED") {
                throw new ApiError(
                    503,
                    "Model analysis service is unavailable"
                );
            } else if (error.code === "ENOTFOUND") {
                throw new ApiError(
                    503,
                    "Model analysis service host not found"
                );
            } else if (error.code === "ETIMEDOUT") {
                throw new ApiError(504, "Model analysis service timed out");
            } else {
                throw new ApiError(
                    500,
                    `Model analysis failed: ${error.message}`
                );
            }
        }
    }

    /**
     * Analyzes video with fallback to mock data if service is unavailable
     * @param {string} videoPath - Path to the video file
     * @param {string} videoId - Unique identifier for the video
     * @param {Object} fallbackData - Mock data to use if service is unavailable
     * @returns {Promise<Object>} Analysis results (real or mock)
     */
    async analyzeVideoWithFallback(videoPath, videoId, fallbackData = null) {
        try {
            // First, check if the service is available
            if (!this.isAvailable()) {
                logger.warn(
                    `LSTM model service not available for video ${videoId}, using fallback data`
                );
                return this.generateFallbackResult(fallbackData);
            }

            // Try to analyze with the LSTM model
            return await this.analyzeVideo(videoPath, videoId);
        } catch (error) {
            logger.warn(
                `LSTM model analysis failed for video ${videoId}, falling back to mock data: ${error.message}`
            );
            return this.generateFallbackResult(fallbackData);
        }
    }

    /**
     * Generates fallback/mock analysis result
     * @param {Object} fallbackData - Optional data to base the mock result on
     * @returns {Object} Mock analysis result
     */
    generateFallbackResult(fallbackData = null) {
        const isLikelyReal = Math.random() > 0.3;
        const baseConfidence = isLikelyReal
            ? 0.75 + Math.random() * 0.2
            : 0.6 + Math.random() * 0.3;
        const confidence = parseFloat(
            Math.min(0.98, Math.max(0.52, baseConfidence)).toFixed(4)
        );
        const prediction = confidence > 0.75 ? "REAL" : "FAKE";
        const processingTime = parseFloat((2.5 + Math.random() * 2).toFixed(2));

        return {
            prediction,
            confidence,
            processingTime,
            model: "LSTM_SIGLIP",
            modelVersion: "fallback",
            status: "COMPLETED",
            timestamp: new Date().toISOString(),
            isMockData: true,
        };
    }
    /**
     * Generates a visualized analysis video by calling the Python service.
     * @param {string} videoPath - Path to the video file to be visualized.
     * @returns {Promise<Stream>} A readable stream of the generated video file.
     */
    async generateVisualAnalysis(videoPath) {
        if (!this.isAvailable()) {
            throw new ApiError(503, "Visual analysis service is not configured");
        }
        if (!fs.existsSync(videoPath)) {
            throw new ApiError(400, "Video file for visualization not found");
        }

        try {
            logger.info(`Requesting visual analysis for video at: ${videoPath}`);

            const formData = new FormData();
            formData.append("video", fs.createReadStream(videoPath));

            const response = await axios.post(
                `${this.serverUrl}/analyze/visualize`,
                formData,
                {
                    headers: {
                        ...formData.getHeaders(),
                        "X-API-Key": this.apiKey,
                    },
                    responseType: 'stream', // CRITICAL: This tells axios to handle the response as a stream
                    timeout: 600000, // 10 minutes timeout for this intensive process
                }
            );

            logger.info(`Successfully received visual analysis stream for: ${videoPath}`);
            return response.data; // This is now a readable stream
        } catch (error) {
            logger.error(`Visual analysis generation failed: ${error.message}`);
            // Handle various errors as in the analyzeVideo method
            if (error.response) {
                const status = error.response.status;
                const message = error.response.data?.detail || "Visual analysis failed";
                throw new ApiError(status, message);
            }
            // ... other error handling ...
            else {
                throw new ApiError(500, `Visual analysis generation failed: ${error.message}`);
            }
        }
    }
}

// Export a singleton instance
export const modelAnalysisService = new ModelAnalysisService();
