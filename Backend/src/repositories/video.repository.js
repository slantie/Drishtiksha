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

    // Update Video details (only filename and description)
    async updateById(videoId, updateData) {
        return prisma.video.update({
            where: { id: videoId },
            data: updateData,
        });
    },

    async deleteById(videoId) {
        return prisma.video.delete({ where: { id: videoId } });
    },

    async getAnalysisStats(timeframe) {
        const now = new Date();
        let startDate;
        switch (timeframe) {
            case "1h":
                startDate = new Date(now.getTime() - 3600000);
                break;
            case "7d":
                startDate = new Date(now.getTime() - 7 * 86400000);
                break;
            case "30d":
                startDate = new Date(now.getTime() - 30 * 86400000);
                break;
            default:
                startDate = new Date(now.getTime() - 86400000);
                break;
        }
        const where = { createdAt: { gte: startDate } };
        const [total, successful, failed, timeAgg, modelBreakdown] =
            await prisma.$transaction([
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
            if (!models[item.model])
                models[item.model] = { total: 0, successful: 0, failed: 0 };
            models[item.model].total += item._count.id;
            if (item.status === "COMPLETED")
                models[item.model].successful += item._count.id;
            else if (item.status === "FAILED")
                models[item.model].failed += item._count.id;
        });
        return {
            timeframe,
            period: { start: startDate, end: now },
            total,
            successful,
            failed,
            avgProcessingTime: timeAgg._avg.processingTime || 0,
            successRate: total > 0 ? successful / total : 0,
            models,
        };
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
                            metrics.frameCount || 
                            metrics.sequenceCount || 
                            metrics.totalFacesDetected ||
                            0,
                        avgConfidence: metrics.finalAverageScore || metrics.averageFaceScore || 0,
                        confidenceStd: metrics.scoreVariance || 0,
                        temporalConsistency:
                            temporalAnalysis?.consistencyScore || null,
                    },
                });
            }
            if (framePredictions?.length > 0) {
                await tx.frameAnalysis.createMany({
                    data: framePredictions.map((frame) => ({
                        analysisId: analysis.id,
                        frameNumber: frame.frameIndex,
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
                            temporalAnalysis.consistencyScore || 0,
                        totalFrames,
                        fakeFrames,
                        realFrames: totalFrames - fakeFrames,
                        avgConfidence: metrics?.finalAverageScore || 0,
                    },
                });
            }
            if (modelInfo)
                await tx.modelInfo.create({
                    data: { analysisId: analysis.id, ...modelInfo },
                });
            if (systemInfo)
                await tx.systemInfo.create({
                    data: { analysisId: analysis.id, ...systemInfo },
                });
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
        const mapStatus = (s) => {
            const statusStr = String(s).toUpperCase();
            if (["OK", "HEALTHY", "RUNNING"].includes(statusStr))
                return "HEALTHY";
            if (["DEGRADED"].includes(statusStr)) return "DEGRADED";
            if (["UNHEALTHY", "ERROR", "FAILED"].includes(statusStr))
                return "UNHEALTHY";
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

    async getServerHealthHistory(serverUrl = null, limit = 50) {
        const where = serverUrl ? { serverUrl } : {};
        return prisma.serverHealth.findMany({
            where,
            orderBy: { createdAt: "desc" },
            take: limit,
        });
    },
};
