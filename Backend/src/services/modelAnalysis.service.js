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

// Server endpoint configuration
const ANALYSIS_ENDPOINTS = {
    QUICK: "/analyze/quick",
    DETAILED: "/analyze/detailed",
    FRAMES: "/analyze/frames",
    VISUALIZE: "/analyze/visualize",
};

// Model mapping for new schema enums
const MODEL_ENUM_MAPPING = {
    siglip_lstm_v1: "SIGLIP_LSTM_V1",
    siglip_lstm_v3: "SIGLIP_LSTM_V3",
    color_cues_lstm_v1: "COLOR_CUES_LSTM_V1",
    // Backward compatibility
    LSTM_SIGLIP: "SIGLIP_LSTM_V1",
};

class ModelAnalysisService {
    constructor() {
        this.serverUrl = process.env.SERVER_URL || "http://localhost:8000";
        this.apiKey = process.env.SERVER_API_KEY;
        this.timeout = 300000; // 5 minutes for analysis
        this.streamTimeout = 600000; // 10 minutes for visualizations

        logger.info(`Model service initialized with URL: ${this.serverUrl}`);

        if (!this.apiKey) {
            logger.warn(
                "SERVER_API_KEY not found in environment variables. Model analysis will be disabled."
            );
        } else {
            logger.info("SERVER_API_KEY found - Model analysis enabled");
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
            const response = await axios.get(`${this.serverUrl}/models/info`, {
                timeout: 10000,
                headers: this.apiKey ? { "X-API-Key": this.apiKey } : {},
            });
            return response.data;
        } catch (error) {
            logger.error(`Failed to get model info: ${error.message}`);
            throw new ApiError(503, "Unable to retrieve model information");
        }
    }

    /**
     * Analyzes a video using the specified analysis type and model
     * @param {string} videoPath - Path to the video file
     * @param {string} analysisType - Type of analysis (QUICK, DETAILED, FRAMES, VISUALIZE)
     * @param {string} model - Model to use for analysis (optional)
     * @param {string} videoId - Unique identifier for the video (optional, for logging)
     * @returns {Promise<Object>} Analysis results
     */
    async analyzeVideo(
        videoPath,
        analysisType = "QUICK",
        model = null,
        videoId = null
    ) {
        if (!this.isAvailable()) {
            throw new ApiError(503, "Model analysis service is not configured");
        }

        const endpoint = ANALYSIS_ENDPOINTS[analysisType];
        if (!endpoint) {
            throw new ApiError(400, `Invalid analysis type: ${analysisType}`);
        }

        if (!fs.existsSync(videoPath)) {
            throw new ApiError(400, "Video file not found");
        }

        try {
            const logId = videoId || path.basename(videoPath);
            logger.info(
                `Starting ${analysisType} analysis for video: ${logId}`
            );
            const startTime = Date.now();

            // Create form data for the request
            const formData = new FormData();
            formData.append("video", fs.createReadStream(videoPath));

            if (model) {
                // Convert model enum to server format if needed
                const serverModel = this._getServerModelName(model);
                formData.append("model", serverModel);
            }

            if (videoId) {
                formData.append("video_id", videoId);
            }

            // Make request to the model server
            const response = await axios.post(
                `${this.serverUrl}${endpoint}`,
                formData,
                {
                    headers: {
                        ...formData.getHeaders(),
                        "X-API-Key": this.apiKey,
                    },
                    timeout: this.timeout,
                    maxContentLength: Infinity,
                    maxBodyLength: Infinity,
                }
            );

            const processingTime = (Date.now() - startTime) / 1000;
            logger.info(
                `${analysisType} analysis completed for video ${logId} in ${processingTime}s`
            );

            return this._processAnalysisResponse(response.data, analysisType);
        } catch (error) {
            const logId = videoId || path.basename(videoPath);
            logger.error(
                `${analysisType} analysis failed for video ${logId}: ${error.message}`
            );

            return this._handleAnalysisError(error, analysisType);
        }
    }

    /**
     * Generates a visualized analysis video by calling the Python service
     * @param {string} videoPath - Path to the video file to be visualized
     * @param {string} model - Model to use for visualization (optional)
     * @param {string} videoId - Unique identifier for the video (optional, for logging)
     * @returns {Promise<Object>} Visualization result with saved file path
     */
    async generateVisualAnalysis(videoPath, model = null, videoId = null) {
        if (!this.isAvailable()) {
            throw new ApiError(
                503,
                "Visual analysis service is not configured"
            );
        }

        if (!fs.existsSync(videoPath)) {
            throw new ApiError(400, "Video file for visualization not found");
        }

        try {
            const logId = videoId || path.basename(videoPath);
            logger.info(`Requesting visual analysis for video: ${logId}`);

            const formData = new FormData();
            formData.append("video", fs.createReadStream(videoPath));

            if (model) {
                const serverModel = this._getServerModelName(model);
                formData.append("model", serverModel);
            }

            if (videoId) {
                formData.append("video_id", videoId);
            }

            const response = await axios.post(
                `${this.serverUrl}${ANALYSIS_ENDPOINTS.VISUALIZE}`,
                formData,
                {
                    headers: {
                        ...formData.getHeaders(),
                        "X-API-Key": this.apiKey,
                    },
                    responseType: "stream",
                    timeout: this.streamTimeout,
                }
            );

            // Generate unique filename for visualization
            const timestamp = Date.now();
            const randomId = Math.floor(Math.random() * 1000000);
            const outputFileName = `visualization-${timestamp}-${randomId}.mp4`;
            const outputPath = path.join(
                process.cwd(),
                "uploads",
                "visualizations",
                outputFileName
            );

            // Ensure directory exists
            const outputDir = path.dirname(outputPath);
            if (!fs.existsSync(outputDir)) {
                fs.mkdirSync(outputDir, { recursive: true });
            }

            // Save the streamed video response
            const writer = fs.createWriteStream(outputPath);
            response.data.pipe(writer);

            return new Promise((resolve, reject) => {
                writer.on("finish", () => {
                    logger.info(`Visualization saved to: ${outputPath}`);
                    resolve({
                        success: true,
                        visualizationPath: outputPath,
                        visualizationUrl: `/uploads/visualizations/${outputFileName}`,
                        message: "Visualization generated successfully",
                    });
                });

                writer.on("error", (error) => {
                    logger.error("Failed to save visualization:", error);
                    reject(
                        new ApiError(500, "Failed to save visualization video")
                    );
                });

                response.data.on("error", (error) => {
                    logger.error("Failed to receive visualization:", error);
                    reject(
                        new ApiError(
                            500,
                            "Failed to receive visualization from server"
                        )
                    );
                });
            });
        } catch (error) {
            const logId = videoId || path.basename(videoPath);
            logger.error(
                `Visual analysis generation failed for ${logId}: ${error.message}`
            );
            return this._handleAnalysisError(error, "VISUALIZE");
        }
    }

    /**
     * Analyzes video with fallback to mock data if service is unavailable
     * @param {string} videoPath - Path to the video file
     * @param {string} videoId - Unique identifier for the video
     * @param {string} analysisType - Type of analysis
     * @param {string} model - Model to use
     * @param {Object} fallbackData - Mock data to use if service is unavailable
     * @returns {Promise<Object>} Analysis results (real or mock)
     */
    async analyzeVideoWithFallback(
        videoPath,
        videoId,
        analysisType = "QUICK",
        model = null,
        fallbackData = null
    ) {
        try {
            if (!this.isAvailable()) {
                logger.warn(
                    `Model service not available for video ${videoId}, using fallback data`
                );
                return this.generateFallbackResult(
                    analysisType,
                    model,
                    fallbackData
                );
            }

            return await this.analyzeVideo(
                videoPath,
                analysisType,
                model,
                videoId
            );
        } catch (error) {
            logger.warn(
                `Model analysis failed for video ${videoId}, falling back to mock data: ${error.message}`
            );
            return this.generateFallbackResult(
                analysisType,
                model,
                fallbackData
            );
        }
    }

    /**
     * Generates fallback/mock analysis result
     * @param {string} analysisType - Type of analysis
     * @param {string} model - Model used
     * @param {Object} fallbackData - Optional data to base the mock result on
     * @returns {Object} Mock analysis result
     */
    generateFallbackResult(
        analysisType = "QUICK",
        model = null,
        fallbackData = null
    ) {
        const isLikelyReal = Math.random() > 0.3;
        const baseConfidence = isLikelyReal
            ? 0.75 + Math.random() * 0.2
            : 0.6 + Math.random() * 0.3;
        const confidence = parseFloat(
            Math.min(0.98, Math.max(0.52, baseConfidence)).toFixed(4)
        );
        const prediction = confidence > 0.75 ? "REAL" : "FAKE";
        const processingTime = parseFloat((2.5 + Math.random() * 2).toFixed(2));

        const result = {
            analysisType,
            prediction,
            confidence,
            is_deepfake: prediction === "FAKE",
            processing_time: processingTime,
            model: model || "SIGLIP_LSTM_V1",
            model_version: "fallback",
            status: "COMPLETED",
            timestamp: new Date().toISOString(),
            isMockData: true,
        };

        // Add type-specific mock data
        if (analysisType === "DETAILED") {
            result.detailed_metrics = {
                frame_consistency: Math.random() * 0.5 + 0.5,
                temporal_coherence: Math.random() * 0.4 + 0.6,
                facial_artifacts: Math.random() * 0.3,
            };
        }

        if (analysisType === "FRAMES") {
            result.frame_analyses = Array.from({ length: 5 }, (_, i) => ({
                frame_number: i * 10,
                confidence: confidence + (Math.random() - 0.5) * 0.1,
                prediction: Math.random() > 0.5 ? "REAL" : "FAKE",
            }));
        }

        return result;
    }

    /**
     * Helper method to get available models
     * @returns {Promise<Array>} List of available models
     */
    async getAvailableModels() {
        try {
            const modelInfo = await this.getModelInfo();
            return (
                modelInfo.available_models || Object.values(MODEL_ENUM_MAPPING)
            );
        } catch (error) {
            logger.warn("Could not retrieve available models:", error.message);
            return Object.values(MODEL_ENUM_MAPPING);
        }
    }

    /**
     * Helper method to validate analysis type
     * @param {string} type - Analysis type to validate
     * @returns {boolean} True if valid
     */
    isValidAnalysisType(type) {
        return Object.keys(ANALYSIS_ENDPOINTS).includes(type);
    }

    /**
     * Helper method to validate model
     * @param {string} model - Model to validate
     * @returns {boolean} True if valid
     */
    isValidModel(model) {
        return (
            Object.values(MODEL_ENUM_MAPPING).includes(model) ||
            Object.keys(MODEL_ENUM_MAPPING).includes(model)
        );
    }

    /**
     * Convert schema model enum to server model name
     * @private
     */
    _getServerModelName(model) {
        // If it's already a server model name, return as-is
        const serverModelNames = Object.keys(MODEL_ENUM_MAPPING);
        if (serverModelNames.includes(model)) {
            return model;
        }

        // Find the server model name for the schema enum
        for (const [serverName, schemaEnum] of Object.entries(
            MODEL_ENUM_MAPPING
        )) {
            if (schemaEnum === model) {
                return serverName;
            }
        }

        // Default fallback
        return "siglip_lstm_v1";
    }

    /**
     * Process and standardize analysis response
     * @private
     */
    _processAnalysisResponse(data, analysisType) {
        const result = {
            analysisType,
            timestamp: new Date().toISOString(),
            ...data,
        };

        // Map model names to schema enums
        if (result.model && MODEL_ENUM_MAPPING[result.model]) {
            result.model = MODEL_ENUM_MAPPING[result.model];
        }

        // Ensure required fields exist
        if (!result.confidence) result.confidence = 0;
        if (typeof result.is_deepfake === "undefined") {
            result.is_deepfake = result.prediction === "FAKE";
        }
        if (!result.processing_time) result.processing_time = 0;

        return result;
    }

    /**
     * Handle analysis errors consistently
     * @private
     */
    _handleAnalysisError(error, analysisType) {
        if (error.response) {
            const status = error.response.status;
            const message =
                error.response.data?.detail ||
                error.response.data?.message ||
                `${analysisType} analysis failed`;
            throw new ApiError(status, message);
        } else if (error.code === "ECONNREFUSED") {
            throw new ApiError(503, "Model analysis service is unavailable");
        } else if (error.code === "ENOTFOUND") {
            throw new ApiError(503, "Model analysis service host not found");
        } else if (error.code === "ETIMEDOUT") {
            throw new ApiError(504, "Model analysis service timed out");
        } else {
            throw new ApiError(
                500,
                `${analysisType} analysis failed: ${error.message}`
            );
        }
    }
}

// Export a singleton instance
export const modelAnalysisService = new ModelAnalysisService();
