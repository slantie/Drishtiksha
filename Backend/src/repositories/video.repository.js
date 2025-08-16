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
                modelInfo: true,
                systemInfo: true,
                errors: true,
            },
        },
    },
};

export const videoRepository = {
    // ... (create, findById, findByIdAndUserId, findAllByUserId, updateStatus, updateAnalysis, deleteById remain the same)

    // ADDED: New method to get aggregated analysis stats.
    // REASON: Moves complex query logic from the controller to the repository layer, improving separation of concerns and performance.
    async getAnalysisStats(timeframe) {
        const now = new Date();
        let startDate;
        switch (timeframe) {
            case "1h":
                startDate = new Date(now.getTime() - 60 * 60 * 1000);
                break;
            case "7d":
                startDate = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
                break;
            case "30d":
                startDate = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
                break;
            case "24h":
            default:
                startDate = new Date(now.getTime() - 24 * 60 * 60 * 1000);
                break;
        }

        const where = { createdAt: { gte: startDate } };

        const [
            totalAnalyses,
            successfulAnalyses,
            failedAnalyses,
            processingTimeAgg,
            modelBreakdown,
        ] = await prisma.$transaction([
            prisma.deepfakeAnalysis.count({ where }),
            prisma.deepfakeAnalysis.count({
                where: { ...where, status: "COMPLETED" },
            }),
            prisma.deepfakeAnalysis.count({
                where: { ...where, status: "FAILED" },
            }),
            prisma.deepfakeAnalysis.aggregate({
                where: {
                    ...where,
                    status: "COMPLETED",
                    processingTime: { not: null },
                },
                _avg: { processingTime: true },
            }),
            prisma.deepfakeAnalysis.groupBy({
                by: ["model", "status"],
                where,
                _count: { id: true },
            }),
        ]);

        const models = {};
        modelBreakdown.forEach((item) => {
            if (!models[item.model]) {
                models[item.model] = { total: 0, successful: 0, failed: 0 };
            }
            models[item.model].total += item._count.id;
            if (item.status === "COMPLETED")
                models[item.model].successful += item._count.id;
            else if (item.status === "FAILED")
                models[item.model].failed += item._count.id;
        });

        return {
            timeframe,
            period: { start: startDate, end: now },
            total: totalAnalyses,
            successful: successfulAnalyses,
            failed: failedAnalyses,
            avgProcessingTime: processingTimeAgg._avg.processingTime || 0,
            successRate:
                totalAnalyses > 0 ? successfulAnalyses / totalAnalyses : 0,
            models,
        };
    },

    // ... (createAnalysisResult, createAnalysisError, storeServerHealth, etc. remain the same)

    // NOTE: The rest of the file (create, findById, etc.) is unchanged from the previous step.
    // For brevity, I'm omitting the duplicated code. Please ensure the new getAnalysisStats method
    // is added to your existing, corrected video.repository.js file.
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

    async updateAnalysis(analysisId, data) {
        return prisma.deepfakeAnalysis.update({
            where: { id: analysisId },
            data,
        });
    },

    async deleteById(videoId) {
        return prisma.video.delete({ where: { id: videoId } });
    },

    async createAnalysisResult(videoId, resultData) {
        const {
            prediction,
            confidence,
            processingTime,
            model,
            modelVersion,
            analysisType,
            metrics,
            framePredictions,
            temporalAnalysis,
            modelInfo,
            systemInfo,
        } = resultData;

        return prisma.$transaction(async (tx) => {
            const analysis = await tx.deepfakeAnalysis.create({
                data: {
                    videoId,
                    prediction,
                    confidence,
                    processingTime,
                    model,
                    modelVersion,
                    analysisType,
                    status: "COMPLETED",
                    timestamp: new Date(),
                },
            });

            if (metrics) {
                await tx.analysisDetails.create({
                    data: {
                        analysisId: analysis.id,
                        frameCount:
                            metrics.frame_count || metrics.sequence_count || 0,
                        avgConfidence: metrics.final_average_score || 0,
                        confidenceStd: metrics.score_variance || 0,
                        temporalConsistency:
                            temporalAnalysis?.consistency_score || null,
                    },
                });
            }

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

            if (temporalAnalysis) {
                const totalFrames = framePredictions?.length || 0;
                const fakeFrames =
                    framePredictions?.filter((p) => p.prediction === "FAKE")
                        .length || 0;
                await tx.temporalAnalysis.create({
                    data: {
                        analysisId: analysis.id,
                        consistencyScore:
                            temporalAnalysis.consistency_score || 0,
                        totalFrames: totalFrames,
                        fakeFrames: fakeFrames,
                        realFrames: totalFrames - fakeFrames,
                        avgConfidence: metrics?.final_average_score || 0,
                    },
                });
            }

            if (modelInfo) {
                await tx.modelInfo.create({
                    data: { analysisId: analysis.id, ...modelInfo },
                });
            }
            if (systemInfo) {
                await tx.systemInfo.create({
                    data: { analysisId: analysis.id, ...systemInfo },
                });
            }

            return analysis;
        });
    },

    async createAnalysisError(videoId, model, analysisType, error) {
        return prisma.$transaction(async (tx) => {
            const analysis = await tx.deepfakeAnalysis.create({
                data: {
                    videoId,
                    model,
                    analysisType,
                    status: "FAILED",
                    errorMessage: error.message,
                    prediction: "REAL",
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

    async storeServerHealth(healthData) {
        const mapStatus = (serverStatus) => {
            const statusStr = String(serverStatus).toUpperCase();
            if (["OK", "HEALTHY", "RUNNING"].includes(statusStr))
                return "HEALTHY";
            if (["DEGRADED"].includes(statusStr)) return "DEGRADED";
            if (["UNHEALTHY", "ERROR", "FAILED"].includes(statusStr))
                return "UNHEALTHY";
            if (["MAINTENANCE"].includes(statusStr)) return "MAINTENANCE";
            return "UNKNOWN";
        };

        return prisma.serverHealth.create({
            data: {
                serverUrl: healthData.service_name
                    ? `${process.env.SERVER_URL} (${healthData.service_name})`
                    : process.env.SERVER_URL || "unknown",
                status: mapStatus(healthData.status),
                availableModels:
                    healthData.models_info?.map((m) => m.name) || [],
                modelStates: healthData.models_info || null,
                gpuInfo: healthData.device_info || null,
                systemResources: healthData.system_info || null,
                lastHealthCheck: new Date(),
                responseTime: healthData.responseTime || null,
                uptime: `${healthData.uptime_seconds || 0}s`,
                version: healthData.version || null,
            },
        });
    },

    async getLatestServerHealth(serverUrl = null) {
        const where = serverUrl ? { serverUrl } : {};
        return prisma.serverHealth.findFirst({
            where,
            orderBy: { createdAt: "desc" },
        });
    },

    async getServerHealthHistory(serverUrl = null, limit = 50) {
        const where = serverUrl ? { serverUrl } : {};
        return prisma.serverHealth.findMany({
            where,
            orderBy: { createdAt: "desc" },
            take: limit,
        });
    },
};
