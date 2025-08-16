// src/services/modelAnalysis.service.js

import axios from "axios";
import FormData from "form-data";
import fs from "fs";
import path from "path";
import { ApiError } from "../utils/ApiError.js";
import logger from "../utils/logger.js";

const ANALYSIS_ENDPOINTS = {
    QUICK: "/analyze",
    FRAMES: "/analyze/frames",
    VISUALIZE_STREAM: "/analyze/visualize",
    VISUALIZE_DOWNLOAD: "/analyze/visualization",
    COMPREHENSIVE: "/analyze/comprehensive",
};

const MONITORING_ENDPOINTS = {
    STATS: "/stats",
};

const MODEL_ENUM_MAPPING = {
    "SIGLIP-LSTM-V1": "SIGLIP_LSTM_V1",
    "SIGLIP-LSTM-V3": "SIGLIP_LSTM_V3",
    "COLOR-CUES-LSTM-V1": "COLOR_CUES_LSTM_V1",
};

class ModelAnalysisService {
    constructor() {
        this.serverUrl = process.env.SERVER_URL;
        this.apiKey = process.env.SERVER_API_KEY;
        this.comprehensiveTimeout = 1200000;
        this.healthTimeout = 20000;

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

    // CHANGED: This is now the single source of truth for server state.
    // REASON: Consolidates all server state fetching into one efficient, comprehensive call.
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

            // Return the full stats object, now including responseTime
            return { ...response.data, responseTime };
        } catch (error) {
            this._handleAnalysisError(error, "STATS");
        }
    }

    // DEPRECATED: This method is no longer needed as getServerStatistics provides all necessary info.
    // getHealthStatus() is now fully replaced.

    async analyzeVideoComprehensive(videoPath, modelName, videoId) {
        if (!this.isAvailable()) {
            throw new ApiError(
                503,
                "Model analysis service is not configured."
            );
        }
        if (!fs.existsSync(videoPath)) {
            throw new ApiError(
                404,
                `Video file not found at path: ${videoPath}`
            );
        }
        const logId = videoId || path.basename(videoPath);
        logger.info(
            `Starting comprehensive analysis for video ${logId} with model ${modelName}`
        );
        try {
            const formData = new FormData();
            formData.append("video", fs.createReadStream(videoPath));
            formData.append("model", modelName);
            formData.append("include_frames", "true");
            formData.append("include_visualization", "true");
            const response = await axios.post(
                `${this.serverUrl}${ANALYSIS_ENDPOINTS.COMPREHENSIVE}`,
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
                    "Comprehensive analysis failed with an invalid response from the model server."
                );
            }
            logger.info(`Comprehensive analysis completed for video ${logId}`);
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

    mapModelNameToEnum(modelName) {
        return MODEL_ENUM_MAPPING[modelName] || null;
    }

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
