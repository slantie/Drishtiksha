// src/services/modelAnalysis.service.js

import axios from "axios";
import FormData from "form-data";
import fs from "fs";
import path from "path";
import http from "http";
import { config } from "dotenv";
import { ApiError } from "../utils/ApiError.js";
import logger from "../utils/logger.js";

config();

const ANALYSIS_ENDPOINTS = {
    QUICK: "/analyze",
    DETAILED: "/analyze/detailed", // Legacy - kept for manual individual requests
    FRAMES: "/analyze/frames", // Legacy - kept for manual individual requests
    VISUALIZE: "/analyze/visualize", // Legacy - kept for manual individual requests
    COMPREHENSIVE: "/analyze/comprehensive", // New default comprehensive endpoint
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
        this.timeout = 300000; // 5 minutes for analysis
        this.streamTimeout = 600000; // 10 minutes for visualizations
        this.comprehensiveTimeout = 900000; // 15 minutes for comprehensive analysis
        this.healthTimeout = 15000; // 15 seconds for health checks

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

    async getHealthStatus() {
        if (!this.isAvailable())
            throw new ApiError(503, "Model service is not configured.");
        try {
            const startTime = Date.now();
            const response = await axios.get(`${this.serverUrl}/`, {
                timeout: this.healthTimeout,
                headers: {
                    "X-API-Key": this.apiKey,
                },
                // Add keep-alive and other connection options
                httpAgent: new http.Agent({
                    keepAlive: true,
                    timeout: this.healthTimeout,
                }),
            });
            const responseTime = Date.now() - startTime;

            // Enhance response with timing information
            const healthData = {
                ...response.data,
                responseTime,
                serverUrl: this.serverUrl,
                timestamp: new Date().toISOString(),
            };

            return healthData;
        } catch (error) {
            logger.error(`Model server health check failed: ${error.message}`);

            // More specific error handling
            if (error.code === "ECONNRESET") {
                throw new ApiError(
                    503,
                    "Connection reset by model analysis service. The service may be restarting."
                );
            } else if (error.code === "ETIMEDOUT") {
                throw new ApiError(
                    504,
                    "Model analysis service health check timed out."
                );
            } else if (error.code === "ECONNREFUSED") {
                throw new ApiError(
                    503,
                    "Model analysis service connection refused. Is the service running?"
                );
            }

            throw new ApiError(503, "Model analysis service is unavailable.");
        }
    }

    async getAvailableModels() {
        if (!this.isAvailable()) {
            throw new ApiError(503, "Model service is not configured.");
        }

        try {
            const healthStatus = await this.getHealthStatus();

            // Extract available models from health status
            const availableModels =
                healthStatus.active_models
                    ?.filter((model) => model.loaded)
                    .map((model) => this.mapModelNameToEnum(model.name))
                    .filter(Boolean) || [];

            logger.info(
                `Available models from server: ${availableModels.join(", ")}`
            );
            return availableModels;
        } catch (error) {
            logger.error(`Failed to get available models: ${error.message}`);

            // Return empty array if we can't reach the server
            // This allows the frontend to gracefully handle the case
            return [];
        }
    }

    async analyzeVideoComprehensive(
        videoPath,
        model,
        videoId,
        includeFrames = true,
        includeVisualization = true
    ) {
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
            `Starting comprehensive analysis for video ${logId} with model ${model} (frames: ${includeFrames}, visualization: ${includeVisualization})`
        );

        try {
            const formData = new FormData();
            formData.append("video", fs.createReadStream(videoPath));
            formData.append("model", model);
            formData.append("include_frames", includeFrames);
            formData.append("include_visualization", includeVisualization);

            const response = await axios.post(
                `${this.serverUrl}${ANALYSIS_ENDPOINTS.COMPREHENSIVE}`,
                formData,
                {
                    headers: {
                        ...formData.getHeaders(),
                        "X-API-Key": this.apiKey,
                    },
                    timeout: this.comprehensiveTimeout, // Use longer timeout for comprehensive analysis
                }
            );

            // The comprehensive response structure is { success, model_used, data: {...} }
            if (!response.data || !response.data.success) {
                throw new ApiError(
                    500,
                    "Comprehensive analysis failed with an invalid response from the model server."
                );
            }

            logger.info(
                `Comprehensive analysis completed for video ${logId} with model ${model}`
            );

            return this._standardizeComprehensiveResponse(response.data);
        } catch (error) {
            this._handleAnalysisError(error, "COMPREHENSIVE", logId);
        }
    }

    async analyzeVideo(videoPath, analysisType, model, videoId) {
        if (!this.isAvailable()) {
            throw new ApiError(
                503,
                "Model analysis service is not configured."
            );
        }

        const endpoint = ANALYSIS_ENDPOINTS[analysisType];
        if (!endpoint) {
            throw new ApiError(400, `Invalid analysis type: ${analysisType}`);
        }
        if (!fs.existsSync(videoPath)) {
            throw new ApiError(
                404,
                `Video file not found at path: ${videoPath}`
            );
        }

        const logId = videoId || path.basename(videoPath);
        logger.info(
            `Starting ${analysisType} analysis for video ${logId} with model ${model}`
        );

        try {
            const formData = new FormData();
            formData.append("video", fs.createReadStream(videoPath));
            formData.append("model", model);

            const response = await axios.post(
                `${this.serverUrl}${endpoint}`,
                formData,
                {
                    headers: {
                        ...formData.getHeaders(),
                        "X-API-Key": this.apiKey,
                    },
                    timeout: this.timeout,
                }
            );

            // The new response structure is { success, model_used, data: {...} }
            if (!response.data || !response.data.success) {
                throw new ApiError(
                    500,
                    "Analysis failed with an invalid response from the model server."
                );
            }

            return this._standardizeResponse(response.data);
        } catch (error) {
            this._handleAnalysisError(error, analysisType, logId);
        }
    }

    async generateVisualAnalysis(videoPath, model, videoId) {
        if (!this.isAvailable()) {
            throw new ApiError(
                503,
                "Visual analysis service is not configured."
            );
        }

        const logId = videoId || path.basename(videoPath);
        logger.info(
            `Requesting visual analysis for video ${logId} with model ${model}`
        );

        try {
            const formData = new FormData();
            formData.append("video", fs.createReadStream(videoPath));
            formData.append("model", model);

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

            const outputFileName = `visualization-${Date.now()}-${Math.round(
                Math.random() * 1e6
            )}.mp4`;
            const outputPath = path.join(
                "uploads",
                "visualizations",
                outputFileName
            );
            const writer = fs.createWriteStream(outputPath);
            response.data.pipe(writer);

            return new Promise((resolve, reject) => {
                writer.on("finish", () => {
                    logger.info(
                        `Visualization for ${logId} saved to: ${outputPath}`
                    );
                    resolve({
                        success: true,
                        visualizationPath: outputPath,
                        visualizationUrl: `/uploads/visualizations/${outputFileName}`,
                    });
                });
                writer.on("error", (err) => {
                    logger.error(
                        `Failed to write visualization stream for ${logId}: ${err.message}`
                    );
                    reject(
                        new ApiError(500, "Failed to save visualization video.")
                    );
                });
            });
        } catch (error) {
            this._handleAnalysisError(error, "VISUALIZE", logId);
        }
    }

    mapModelNameToEnum(modelName) {
        return MODEL_ENUM_MAPPING[modelName] || null;
    }

    _standardizeResponse(response) {
        // Standardize the new response { success, model_used, data: {...} }
        const { model_used, data } = response;
        return {
            model: this.mapModelNameToEnum(model_used),
            modelVersion: model_used,
            prediction: data.prediction || data.overall_prediction,
            confidence: data.confidence || data.overall_confidence,
            processingTime: data.processing_time,
            metrics: data.metrics, // For DETAILED
            framePredictions: data.frame_predictions, // For FRAMES
            temporalAnalysis: data.temporal_analysis, // For FRAMES
            note: data.note,
        };
    }

    _standardizeComprehensiveResponse(response) {
        // Standardize the comprehensive response { success, model_used, data: {...} }
        const { model_used, data, timestamp } = response;

        const standardized = {
            // Basic analysis data
            model: this.mapModelNameToEnum(model_used),
            modelVersion: model_used,
            prediction: data.prediction,
            confidence: data.confidence,
            processingTime: data.processing_time,
            metrics: data.metrics,
            note: data.note,

            // Processing breakdown for transparency
            processingBreakdown: data.processing_breakdown,

            // Frame analysis data (if included)
            framePredictions: null,
            temporalAnalysis: null,

            // Visualization data (if included)
            visualizationGenerated: data.visualization_generated || false,
            visualizationFilename: data.visualization_filename,

            // Enhanced monitoring information from actual server response
            modelInfo: {
                model_name: model_used,
                version: "v1.0", // Default since not provided by server
                architecture: "LSTM", // Inferred from model name
                device: "cuda", // Default assumption
                batch_size: null,
                num_frames: data.metrics?.frame_count || null,
                model_size: null,
                load_time: null,
                memory_usage: null,
            },
            systemInfo: {
                gpu_memory_used: null,
                gpu_memory_total: null,
                processing_device: "cuda",
                cuda_available: true,
                cuda_version: null,
                system_memory_used: null,
                system_memory_total: null,
                cpu_usage: null,
                load_balancing_info: null,
                server_version: null,
                python_version: null,
                torch_version: null,
            },
            serverInfo: {
                version: null,
                uptime: null,
                request_id: null,
            },
            requestId: timestamp || new Date().toISOString(),
        };

        // Include frame analysis if present
        if (data.frames_analysis) {
            standardized.framePredictions =
                data.frames_analysis.frame_predictions;
            standardized.temporalAnalysis =
                data.frames_analysis.temporal_analysis;
        }

        return standardized;
    }

    _handleAnalysisError(error, analysisType, logId) {
        logger.error(
            `Analysis failed for ${logId} (${analysisType}): ${error.message}`
        );
        if (error.response) {
            const { status, data } = error.response;
            const message =
                data?.message || "Analysis failed on the model server.";
            logger.error(
                `Model server responded with status ${status}: ${message}`
            );
            throw new ApiError(status, message, data?.details);
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
            error.code === "ECONNABORTED" ||
            error.message.includes("timeout")
        ) {
            throw new ApiError(
                504,
                "Request to model analysis service timed out."
            );
        } else if (error.code === "ENOTFOUND") {
            throw new ApiError(
                503,
                "Model analysis service host not found. Check SERVER_URL configuration."
            );
        } else {
            throw new ApiError(
                500,
                `An unknown error occurred: ${error.message}`
            );
        }
    }
}

export const modelAnalysisService = new ModelAnalysisService();
