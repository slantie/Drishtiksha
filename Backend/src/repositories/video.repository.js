// src/repositories/video.repository.js

import prisma from "../config/database.js";

const videoWithDetails = {
    include: {
        user: {
            select: { id: true, firstName: true, lastName: true, email: true },
        },
        analyses: {
            orderBy: { createdAt: "desc" },
            include: {
                analysisDetails: true,
                frameAnalysis: { orderBy: { frameNumber: "asc" } },
                temporalAnalysis: true,
            },
        },
    },
};

export const videoRepository = {
    async create(videoData) {
        return prisma.video.create({ data: videoData });
    },

    async findById(videoId) {
        return prisma.video.findUnique({
            where: { id: videoId },
            ...videoWithDetails,
        });
    },

    async findByIdAndUserId(videoId, userId) {
        return prisma.video.findFirst({
            where: { id: videoId, userId },
            ...videoWithDetails,
        });
    },

    async findAllByUserId(userId) {
        return prisma.video.findMany({
            where: { userId },
            ...videoWithDetails,
            orderBy: { createdAt: "desc" },
        });
    },

    async updateStatus(videoId, status) {
        return prisma.video.update({
            where: { id: videoId },
            data: { status },
        });
    },

    async deleteById(videoId) {
        return prisma.video.delete({ where: { id: videoId } });
    },

    /**
     * Creates a complete analysis result in a single database transaction.
     * @param {string} videoId - The ID of the video being analyzed.
     * @param {object} resultData - The standardized result from modelAnalysisService.
     */
    async createAnalysisResult(videoId, resultData) {
        const {
            prediction,
            confidence,
            processingTime,
            model,
            modelVersion,
            analysisType,
            visualizedUrl,
            metrics,
            framePredictions,
            temporalAnalysis,
            modelInfo,
            systemInfo,
            serverInfo,
            requestId,
        } = resultData;

        // Use a Prisma transaction to ensure all related data is created successfully.
        return prisma.$transaction(async (tx) => {
            // 1. Create the main DeepfakeAnalysis record
            const analysis = await tx.deepfakeAnalysis.create({
                data: {
                    videoId,
                    prediction,
                    confidence,
                    processingTime,
                    model,
                    modelVersion,
                    analysisType,
                    visualizedUrl,
                    status: "COMPLETED",
                    timestamp: new Date(),
                },
            });

            // 2. If detailed metrics exist, create the related AnalysisDetails record
            if (metrics) {
                await tx.analysisDetails.create({
                    data: {
                        analysisId: analysis.id,
                        frameCount:
                            metrics.sequence_count || metrics.frame_count || 0,
                        avgConfidence: metrics.final_average_score || 0,
                        confidenceStd: metrics.score_variance || 0,
                        temporalConsistency:
                            temporalAnalysis?.consistency_score || null,
                    },
                });
            }

            // 3. If frame-by-frame predictions exist, create them
            if (framePredictions && framePredictions.length > 0) {
                await tx.frameAnalysis.createMany({
                    data: framePredictions.map((frame) => ({
                        analysisId: analysis.id,
                        frameNumber: frame.frame_index,
                        confidence: frame.score,
                        prediction: frame.prediction,
                    })),
                });
            }

            // 4. If temporal analysis exists, create it
            if (temporalAnalysis) {
                await tx.temporalAnalysis.create({
                    data: {
                        analysisId: analysis.id,
                        consistencyScore:
                            temporalAnalysis.consistency_score || 0,
                        patternDetection:
                            temporalAnalysis.pattern_detection || null,
                        anomalyFrames: temporalAnalysis.anomaly_frames || [],
                        confidenceTrend:
                            temporalAnalysis.confidence_trend || null,
                        totalFrames: temporalAnalysis.total_frames || 0,
                        fakeFrames: temporalAnalysis.fake_frames || 0,
                        realFrames: temporalAnalysis.real_frames || 0,
                        avgConfidence: temporalAnalysis.avg_confidence || 0,
                    },
                });
            }

            // 5. If model information exists, create it
            if (modelInfo && Object.keys(modelInfo).length > 0) {
                await tx.modelInfo.create({
                    data: {
                        analysisId: analysis.id,
                        modelName:
                            modelInfo.model_name || modelVersion || model,
                        version: modelInfo.version || modelVersion || "unknown",
                        architecture: modelInfo.architecture || "unknown",
                        device: modelInfo.device || "unknown",
                        batchSize: modelInfo.batch_size || null,
                        numFrames: modelInfo.num_frames || null,
                        modelSize: modelInfo.model_size || null,
                        loadTime: modelInfo.load_time || null,
                        memoryUsage: modelInfo.memory_usage || null,
                    },
                });
            }

            // 6. If system information exists, create it
            if (systemInfo && Object.keys(systemInfo).length > 0) {
                await tx.systemInfo.create({
                    data: {
                        analysisId: analysis.id,
                        gpuMemoryUsed: systemInfo.gpu_memory_used || null,
                        gpuMemoryTotal: systemInfo.gpu_memory_total || null,
                        processingDevice:
                            systemInfo.processing_device ||
                            systemInfo.device ||
                            null,
                        cudaAvailable: systemInfo.cuda_available || null,
                        cudaVersion: systemInfo.cuda_version || null,
                        systemMemoryUsed: systemInfo.system_memory_used || null,
                        systemMemoryTotal:
                            systemInfo.system_memory_total || null,
                        cpuUsage: systemInfo.cpu_usage || null,
                        loadBalancingInfo:
                            systemInfo.load_balancing_info || null,
                        serverVersion:
                            systemInfo.server_version ||
                            serverInfo?.version ||
                            null,
                        pythonVersion: systemInfo.python_version || null,
                        torchVersion: systemInfo.torch_version || null,
                        requestId: requestId || null,
                    },
                });
            }

            return analysis;
        });
    },

    /**
     * Creates a failed analysis record and a detailed error log in a transaction.
     * @param {string} videoId - The ID of the video.
     * @param {string} model - The model enum value.
     * @param {string} analysisType - The type of analysis that failed.
     * @param {Error} error - The caught error object.
     */
    async createAnalysisError(videoId, model, analysisType, error) {
        return prisma.$transaction(async (tx) => {
            const analysis = await tx.deepfakeAnalysis.create({
                data: {
                    videoId,
                    model,
                    analysisType,
                    status: "FAILED",
                    errorMessage: error.message,
                    prediction: "REAL", // Default value
                    confidence: 0,
                },
            });

            await tx.analysisError.create({
                data: {
                    analysisId: analysis.id,
                    errorType: error.name || "AnalysisError",
                    errorMessage: error.message,
                    serverResponse: error.details || { stack: error.stack },
                },
            });

            return analysis;
        });
    },

    /**
     * Store server health information for monitoring
     * @param {Object} healthData - Server health information
     */
    async storeServerHealth(healthData) {
        const {
            serverUrl,
            status,
            availableModels,
            modelStates,
            loadMetrics,
            gpuInfo,
            systemResources,
            responseTime,
            errorMessage,
            requestCount,
            avgProcessingTime,
            uptime,
            version,
        } = healthData;

        // Map server status to enum values
        const mapStatus = (serverStatus) => {
            if (!serverStatus) return "UNKNOWN";
            const statusStr = String(serverStatus).toLowerCase();

            switch (statusStr) {
                case "ok":
                case "healthy":
                case "active":
                    return "HEALTHY";
                case "degraded":
                case "partial":
                    return "DEGRADED";
                case "unhealthy":
                case "error":
                case "failed":
                    return "UNHEALTHY";
                case "maintenance":
                    return "MAINTENANCE";
                default:
                    return "UNKNOWN";
            }
        };

        return prisma.serverHealth.create({
            data: {
                serverUrl: serverUrl || process.env.SERVER_URL || "unknown",
                status: mapStatus(status),
                availableModels: availableModels || [],
                modelStates: modelStates || null,
                loadMetrics: loadMetrics || null,
                gpuInfo: gpuInfo || null,
                systemResources: systemResources || null,
                lastHealthCheck: new Date(),
                responseTime: responseTime || null,
                errorMessage: errorMessage || null,
                requestCount: requestCount || null,
                avgProcessingTime: avgProcessingTime || null,
                uptime: uptime || null,
                version: version || null,
            },
        });
    },

    /**
     * Get the latest server health information
     * @param {string} serverUrl - Optional server URL filter
     */
    async getLatestServerHealth(serverUrl = null) {
        const where = serverUrl ? { serverUrl } : {};

        return prisma.serverHealth.findFirst({
            where,
            orderBy: { createdAt: "desc" },
        });
    },

    /**
     * Get server health history for monitoring dashboard
     * @param {string} serverUrl - Optional server URL filter
     * @param {number} limit - Number of records to return (default 50)
     */
    async getServerHealthHistory(serverUrl = null, limit = 50) {
        const where = serverUrl ? { serverUrl } : {};

        return prisma.serverHealth.findMany({
            where,
            orderBy: { createdAt: "desc" },
            take: limit,
        });
    },
};
