// src/api/monitoring/monitoring.controller.js

import { PrismaClient } from "@prisma/client";
import { videoRepository } from "../../repositories/video.repository.js";
import { modelAnalysisService } from "../../services/modelAnalysis.service.js";
import { ApiResponse } from "../../utils/ApiResponse.js";
import { asyncHandler } from "../../utils/asyncHandler.js";
import logger from "../../utils/logger.js";

const prisma = new PrismaClient();

/**
 * Get current server health status
 */
export const getServerHealth = asyncHandler(async (req, res) => {
    try {
        const healthStatus = await modelAnalysisService.getHealthStatus();

        // Store current health check in database
        await videoRepository.storeServerHealth({
            serverUrl: healthStatus.serverUrl,
            status: "HEALTHY", // Always use HEALTHY for successful responses
            availableModels:
                healthStatus.active_models?.map((m) => m.name) || [],
            modelStates: healthStatus.active_models || null,
            loadMetrics: healthStatus.load_metrics || null,
            gpuInfo: healthStatus.gpu_info || null,
            systemResources: healthStatus.system_resources || null,
            responseTime: healthStatus.responseTime,
            uptime: healthStatus.uptime,
            version: healthStatus.version,
        });

        res.status(200).json(
            new ApiResponse(
                200,
                {
                    status: "HEALTHY",
                    server: {
                        url: healthStatus.serverUrl,
                        version: healthStatus.version,
                        uptime: healthStatus.uptime,
                        responseTime: healthStatus.responseTime,
                    },
                    models: healthStatus.active_models || [],
                    resources: {
                        gpu: healthStatus.gpu_info,
                        system: healthStatus.system_resources,
                        load: healthStatus.load_metrics,
                    },
                    timestamp: new Date().toISOString(),
                },
                "Server health retrieved successfully"
            )
        );
    } catch (error) {
        logger.error(`Failed to get server health: ${error.message}`);

        // Store failed health check
        try {
            await videoRepository.storeServerHealth({
                serverUrl: process.env.SERVER_URL,
                status: "UNHEALTHY",
                availableModels: [],
                errorMessage: error.message,
                responseTime: null,
            });
        } catch (storeError) {
            logger.error(
                `Failed to store error health status: ${storeError.message}`
            );
        }

        res.status(503).json(
            new ApiResponse(
                503,
                null,
                `Server health check failed: ${error.message}`
            )
        );
    }
});

/**
 * Get server health history for monitoring dashboard
 */
export const getServerHealthHistory = asyncHandler(async (req, res) => {
    const { limit = 50, serverUrl } = req.query;

    try {
        const healthHistory = await videoRepository.getServerHealthHistory(
            serverUrl,
            parseInt(limit)
        );

        // Calculate some basic metrics
        const metrics = {
            totalChecks: healthHistory.length,
            healthyChecks: healthHistory.filter((h) => h.status === "HEALTHY")
                .length,
            avgResponseTime:
                healthHistory
                    .filter((h) => h.responseTime)
                    .reduce((sum, h) => sum + h.responseTime, 0) /
                    healthHistory.filter((h) => h.responseTime).length || 0,
            lastCheck: healthHistory[0]?.lastHealthCheck,
            uptime: healthHistory[0]?.uptime,
        };

        res.status(200).json(
            new ApiResponse(
                200,
                {
                    history: healthHistory,
                    metrics,
                    totalRecords: healthHistory.length,
                },
                "Server health history retrieved successfully"
            )
        );
    } catch (error) {
        logger.error(`Failed to get health history: ${error.message}`);
        res.status(500).json(
            new ApiResponse(
                500,
                null,
                `Failed to retrieve health history: ${error.message}`
            )
        );
    }
});

/**
 * Get monitoring statistics for analysis performance
 */
export const getAnalysisStats = asyncHandler(async (req, res) => {
    const { timeframe = "24h", model } = req.query;

    try {
        // Calculate date range based on timeframe
        const now = new Date();
        let startDate;

        switch (timeframe) {
            case "1h":
                startDate = new Date(now.getTime() - 60 * 60 * 1000);
                break;
            case "24h":
                startDate = new Date(now.getTime() - 24 * 60 * 60 * 1000);
                break;
            case "7d":
                startDate = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
                break;
            case "30d":
                startDate = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
                break;
            default:
                startDate = new Date(now.getTime() - 24 * 60 * 60 * 1000);
        }

        // Get analysis statistics from database using Prisma
        const where = {
            createdAt: {
                gte: startDate,
            },
        };

        if (model) {
            where.model = model;
        }

        // Get actual analysis counts and metrics
        const [
            totalAnalyses,
            successfulAnalyses,
            failedAnalyses,
            processingTimes,
            modelBreakdown,
            recentHealthChecks,
        ] = await Promise.all([
            // Total analyses count
            prisma.deepfakeAnalysis.count({ where }),

            // Successful analyses count
            prisma.deepfakeAnalysis.count({
                where: { ...where, status: "COMPLETED" },
            }),

            // Failed analyses count
            prisma.deepfakeAnalysis.count({
                where: { ...where, status: "FAILED" },
            }),

            // Get processing times for average calculation
            prisma.deepfakeAnalysis.findMany({
                where: { ...where, processingTime: { not: null } },
                select: { processingTime: true },
            }),

            // Get breakdown by model
            prisma.deepfakeAnalysis.groupBy({
                by: ["model", "status"],
                where,
                _count: { id: true },
            }),

            // Get recent health checks
            videoRepository.getServerHealthHistory(null, 10),
        ]);

        // Calculate average processing time
        const avgProcessingTime =
            processingTimes.length > 0
                ? processingTimes.reduce(
                      (sum, analysis) => sum + (analysis.processingTime || 0),
                      0
                  ) / processingTimes.length
                : 0;

        // Build model breakdown
        const models = {};
        modelBreakdown.forEach((item) => {
            if (!models[item.model]) {
                models[item.model] = { total: 0, successful: 0, failed: 0 };
            }
            models[item.model].total += item._count.id;
            if (item.status === "COMPLETED") {
                models[item.model].successful += item._count.id;
            } else if (item.status === "FAILED") {
                models[item.model].failed += item._count.id;
            }
        });

        // Calculate server metrics
        const healthyChecks = recentHealthChecks.filter(
            (h) => h.status === "HEALTHY"
        ).length;
        const avgServerResponseTime =
            recentHealthChecks
                .filter((h) => h.responseTime)
                .reduce((sum, h) => sum + h.responseTime, 0) /
                recentHealthChecks.filter((h) => h.responseTime).length || 0;

        const stats = {
            timeframe,
            period: {
                start: startDate,
                end: now,
            },
            analysis: {
                total: totalAnalyses,
                successful: successfulAnalyses,
                failed: failedAnalyses,
                avgProcessingTime: Math.round(avgProcessingTime * 100) / 100,
                successRate:
                    totalAnalyses > 0
                        ? Math.round(
                              (successfulAnalyses / totalAnalyses) * 10000
                          ) / 100
                        : 0,
                models,
            },
            server: {
                healthChecks: recentHealthChecks,
                avgResponseTime: Math.round(avgServerResponseTime * 100) / 100,
                uptime:
                    healthyChecks > 0
                        ? `${
                              Math.round(
                                  (healthyChecks / recentHealthChecks.length) *
                                      10000
                              ) / 100
                          }%`
                        : "0%",
                totalHealthChecks: recentHealthChecks.length,
                healthyChecks,
            },
        };

        res.status(200).json(
            new ApiResponse(
                200,
                stats,
                `Analysis statistics for ${timeframe} retrieved successfully`
            )
        );
    } catch (error) {
        logger.error(`Failed to get analysis stats: ${error.message}`);
        res.status(500).json(
            new ApiResponse(
                500,
                null,
                `Failed to retrieve analysis statistics: ${error.message}`
            )
        );
    }
});

/**
 * Get model performance metrics
 */
export const getModelMetrics = asyncHandler(async (req, res) => {
    try {
        const healthStatus = await modelAnalysisService.getHealthStatus();

        // Get performance data for each model from database
        const modelPerformanceData = await Promise.all(
            (healthStatus.active_models || []).map(async (model) => {
                const modelEnum = modelAnalysisService.mapModelNameToEnum(
                    model.name
                );

                // Get last 24 hours of analysis data for this model
                const last24Hours = new Date(Date.now() - 24 * 60 * 60 * 1000);

                const [
                    totalAnalyses,
                    successfulAnalyses,
                    failedAnalyses,
                    avgProcessingTime,
                ] = await Promise.all([
                    prisma.deepfakeAnalysis.count({
                        where: {
                            model: modelEnum,
                            createdAt: { gte: last24Hours },
                        },
                    }),
                    prisma.deepfakeAnalysis.count({
                        where: {
                            model: modelEnum,
                            status: "COMPLETED",
                            createdAt: { gte: last24Hours },
                        },
                    }),
                    prisma.deepfakeAnalysis.count({
                        where: {
                            model: modelEnum,
                            status: "FAILED",
                            createdAt: { gte: last24Hours },
                        },
                    }),
                    prisma.deepfakeAnalysis.aggregate({
                        where: {
                            model: modelEnum,
                            processingTime: { not: null },
                            createdAt: { gte: last24Hours },
                        },
                        _avg: { processingTime: true },
                    }),
                ]);

                const successRate =
                    totalAnalyses > 0
                        ? Math.round(
                              (successfulAnalyses / totalAnalyses) * 10000
                          ) / 100
                        : 100;

                return {
                    name: model.name,
                    status: model.loaded ? "ACTIVE" : "INACTIVE",
                    device: "cuda", // Default assumption
                    memoryUsage: "unknown", // Not provided by simple health endpoint
                    loadTime: null, // Not provided by simple health endpoint
                    description: model.description || null,
                    performance: {
                        avgProcessingTime:
                            Math.round(
                                (avgProcessingTime._avg.processingTime || 0) *
                                    100
                            ) / 100,
                        successRate,
                        totalAnalyses,
                        successfulAnalyses,
                        failedAnalyses,
                        period: "24h",
                    },
                };
            })
        );

        // Calculate overall metrics
        const totalModels = modelPerformanceData.length;
        const activeModels = modelPerformanceData.filter(
            (m) => m.status === "ACTIVE"
        ).length;
        const overallTotalAnalyses = modelPerformanceData.reduce(
            (sum, m) => sum + m.performance.totalAnalyses,
            0
        );
        const overallSuccessfulAnalyses = modelPerformanceData.reduce(
            (sum, m) => sum + m.performance.successfulAnalyses,
            0
        );
        const overallSuccessRate =
            overallTotalAnalyses > 0
                ? Math.round(
                      (overallSuccessfulAnalyses / overallTotalAnalyses) * 10000
                  ) / 100
                : 100;

        res.status(200).json(
            new ApiResponse(
                200,
                {
                    models: modelPerformanceData,
                    summary: {
                        totalModels,
                        activeModels,
                        inactiveModels: totalModels - activeModels,
                        overallPerformance: {
                            totalAnalyses: overallTotalAnalyses,
                            successfulAnalyses: overallSuccessfulAnalyses,
                            successRate: overallSuccessRate,
                            period: "24h",
                        },
                    },
                    serverInfo: {
                        url: healthStatus.serverUrl,
                        responseTime: healthStatus.responseTime,
                    },
                    timestamp: new Date().toISOString(),
                },
                "Model metrics retrieved successfully"
            )
        );
    } catch (error) {
        logger.error(`Failed to get model metrics: ${error.message}`);
        res.status(500).json(
            new ApiResponse(
                500,
                null,
                `Failed to retrieve model metrics: ${error.message}`
            )
        );
    }
});
