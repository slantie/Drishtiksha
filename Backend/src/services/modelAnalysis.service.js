// src/services/modelAnalysis.service.js

import axios from "axios";
import FormData from "form-data";
import fs from "fs";
import path from "path";
import { ApiError } from "../utils/ApiError.js";
import logger from "../utils/logger.js";

// UPDATED: Added the new AUDIO endpoint
const ANALYSIS_ENDPOINTS = {
    QUICK: "/analyze",
    FRAMES: "/analyze/frames",
    VISUALIZE_STREAM: "/analyze/visualize",
    VISUALIZE_DOWNLOAD: "/analyze/visualization",
    COMPREHENSIVE: "/analyze/comprehensive",
    AUDIO: "/analyze/audio", // NEW
};

const MONITORING_ENDPOINTS = {
    STATS: "/stats",
};

class ModelAnalysisService {
    constructor() {
        this.serverUrl = process.env.SERVER_URL;
        this.apiKey = process.env.SERVER_API_KEY;
        this.comprehensiveTimeout = 1200000; // 20 minutes
        this.healthTimeout = 20000; // 20 seconds

        if (!this.apiKey) {
            logger.warn(
                "SERVER_API_KEY not found. Model analysis service is disabled."
            );
        } else {
            logger.info(
                `Model analysis service initialized for URL: ${this.serverUrl}`
            );
        }
    }

    isAvailable() {
        return Boolean(this.apiKey && this.serverUrl);
    }

    async getServerStatistics() {
        if (!this.isAvailable()) {
            throw new ApiError(503, "Model service is not configured.");
        }
        try {
            const startTime = Date.now();
            const response = await axios.get(
                `${this.serverUrl}${MONITORING_ENDPOINTS.STATS}`,
                {
                    timeout: this.healthTimeout,
                    headers: { "X-API-Key": this.apiKey },
                }
            );
            const responseTime = Date.now() - startTime;
            return { ...response.data, responseTime };
        } catch (error) {
            this._handleAnalysisError(error, "STATS");
        }
    }

    /**
     * REFACTORED: This is now the primary, generic analysis method.
     * It intelligently routes the analysis request to the correct AI server endpoint
     * based on the media type.
     *
     * @param {string} mediaPath - The local path to the media file.
     * @param {string} mediaType - The type of media ('VIDEO', 'AUDIO', 'IMAGE').
     * @param {string} modelName - The name of the model to use.
     * @param {string} mediaId - The ID of the media record.
     * @param {string} userId - The ID of the user.
     * @returns {Promise<object>} The analysis result from the AI server.
     */
    async analyzeMediaComprehensive(
        mediaPath,
        mediaType,
        modelName,
        mediaId,
        userId
    ) {
        if (!this.isAvailable()) {
            throw new ApiError(
                503,
                "Model analysis service is not configured."
            );
        }
        if (!fs.existsSync(mediaPath)) {
            throw new ApiError(
                404,
                `Media file not found at path: ${mediaPath}`
            );
        }

        const logId = mediaId || path.basename(mediaPath);
        logger.info(
            `Starting comprehensive analysis for ${mediaType} ${logId} with model ${modelName}`
        );

        // --- NEW: Dynamic Endpoint and Form Field Selection ---
        let endpoint;
        let formFieldName;

        switch (mediaType) {
            case "VIDEO":
                endpoint = ANALYSIS_ENDPOINTS.COMPREHENSIVE;
                formFieldName = "video"; // The AI server expects the field to be named 'video'
                break;
            case "AUDIO":
                endpoint = ANALYSIS_ENDPOINTS.AUDIO;
                formFieldName = "video"; // The audio endpoint also expects the field 'video'
                break;
            case "IMAGE":
                // Placeholder for when image analysis is added
                // endpoint = ANALYSIS_ENDPOINTS.IMAGE;
                // formFieldName = 'image';
                throw new ApiError(
                    501,
                    "Image analysis is not yet implemented."
                );
            default:
                throw new ApiError(
                    400,
                    `Unsupported media type for analysis: ${mediaType}`
                );
        }

        try {
            const formData = new FormData();
            formData.append(formFieldName, fs.createReadStream(mediaPath));
            formData.append("model", modelName);

            // Pass context to the Python server for logging and progress events
            if (mediaId) formData.append("video_id", mediaId);
            if (userId) formData.append("user_id", userId);

            const response = await axios.post(
                `${this.serverUrl}${endpoint}`,
                formData,
                {
                    headers: {
                        ...formData.getHeaders(),
                        "X-API-Key": this.apiKey,
                    },
                    timeout: this.comprehensiveTimeout,
                }
            );

            if (!response.data || !response.data.success) {
                throw new ApiError(
                    500,
                    `Comprehensive analysis failed with an invalid response from the model server.`
                );
            }

            logger.info(
                `Comprehensive analysis completed for ${mediaType} ${logId}`
            );
            return response.data;
        } catch (error) {
            this._handleAnalysisError(error, "COMPREHENSIVE", logId);
        }
    }

    async downloadVisualization(filename) {
        if (!this.isAvailable()) {
            throw new ApiError(503, "Model service is not configured.");
        }
        try {
            const response = await axios.get(
                `${this.serverUrl}${ANALYSIS_ENDPOINTS.VISUALIZE_DOWNLOAD}/${filename}`,
                {
                    headers: { "X-API-Key": this.apiKey },
                    responseType: "stream",
                }
            );
            return response.data;
        } catch (error) {
            this._handleAnalysisError(error, "VISUALIZE_DOWNLOAD", filename);
        }
    }

    // This function maps the server's statistics to our Prisma schema structure.
    // It remains highly valuable and does not need changes.
    mapServerStatsToDbSchema(stats, modelName) {
        const modelInfoFromServer =
            stats.models_info?.find((m) => m.name === modelName) || {};
        const systemInfoFromServer = stats.system_info || {};
        const deviceInfoFromServer = stats.device_info || {};
        const modelInfo = {
            modelName: modelInfoFromServer.name || modelName,
            version: stats.version || "unknown",
            architecture: modelInfoFromServer.class_name || "unknown",
            device: modelInfoFromServer.device || "unknown",
            memoryUsage: `${modelInfoFromServer.memory_usage_mb || 0}MB`,
        };
        const systemInfo = {
            gpuMemoryUsed: `${deviceInfoFromServer.used_memory || 0}GB`,
            gpuMemoryTotal: `${deviceInfoFromServer.total_memory || 0}GB`,
            processingDevice: deviceInfoFromServer.name || "unknown",
            cudaAvailable: deviceInfoFromServer.type === "cuda",
            cudaVersion: deviceInfoFromServer.cuda_version || null,
            systemMemoryUsed: `${systemInfoFromServer.used_ram || 0}GB`,
            systemMemoryTotal: `${systemInfoFromServer.total_ram || 0}GB`,
            serverVersion: stats.version || "unknown",
            pythonVersion: systemInfoFromServer.python_version || null,
            torchVersion: stats.configuration?.torch_version || null,
            requestId: stats.timestamp || new Date().toISOString(),
        };
        return { modelInfo, systemInfo };
    }

    // The generic error handler is robust and does not need changes.
    _handleAnalysisError(error, analysisType, logId = "") {
        logger.error(
            `Analysis Service Error for ${logId} (${analysisType}): ${error.message}`
        );
        if (error.response) {
            const { status, data } = error.response;
            const message =
                data?.error?.message ||
                data?.message ||
                "Analysis failed on the model server.";
            logger.error(
                `Model server responded with status ${status}: ${JSON.stringify(
                    data
                )}`
            );
            throw new ApiError(status, message, data?.details || data);
        } else if (error.code === "ECONNREFUSED") {
            throw new ApiError(
                503,
                "Model analysis service is unavailable. Check if the service is running."
            );
        } else if (error.code === "ECONNRESET") {
            throw new ApiError(
                503,
                "Connection reset by model analysis service. The service may be overloaded or restarting."
            );
        } else if (
            error.code === "ETIMEDOUT" ||
            error.code === "ECONNABORTED"
        ) {
            throw new ApiError(
                504,
                "Request to model analysis service timed out."
            );
        } else {
            throw new ApiError(
                500,
                `An unknown error occurred during analysis: ${error.message}`
            );
        }
    }
}

export const modelAnalysisService = new ModelAnalysisService();
