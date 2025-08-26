// src/repositories/media.repository.js

import prisma from "../config/database.js";

// RENAMED: from videoWithDetails to mediaWithDetails.
// UPDATED: This object now includes relations for ALL possible media types and analysis types.
// This allows us to fetch a media item and all its related data in a single, efficient query,
// regardless of whether it's a video, image, or audio file.
const mediaWithDetails = {
    include: {
        user: {
            select: { id: true, firstName: true, lastName: true, email: true },
        },
        // Include all possible metadata relations. Only one will be non-null.
        videoMetadata: true,
        imageMetadata: true,
        audioMetadata: true,
        analyses: {
            orderBy: { createdAt: "desc" },
            include: {
                analysisDetails: true,
                frameAnalysis: { orderBy: { frameNumber: "asc" } },
                temporalAnalysis: true,
                // NEW: Include the new audio analysis data.
                audioAnalysis: true,
                modelInfo: true,
                systemInfo: true,
                errors: true,
            },
        },
    },
};

// RENAMED: from videoRepository to mediaRepository.
export const mediaRepository = {
    // UPDATED: Now creates a 'Media' record.
    async create(mediaData) {
        return prisma.media.create({ data: mediaData });
    },

    // NEW: A dedicated function to create the correct metadata record based on mediaType.
    // This abstracts the logic from the service layer, keeping it clean.
    async createMetadata(mediaId, mediaType, metadata) {
        switch (mediaType) {
            case "VIDEO":
                return prisma.videoMetadata.create({
                    data: {
                        mediaId,
                        duration: metadata.duration,
                        width: metadata.width,
                        height: metadata.height,
                        // CORRECTED: 'frameRate' is now 'fps' to match the schema
                        fps: metadata.frameRate || null, 
                        codec: metadata.codec || null,
                        // ENABLED: These fields will now work correctly
                        bitrate: metadata.bitrate || null,
                        resolution: metadata.resolution || null,
                    },
                });
            case "IMAGE":
                return prisma.imageMetadata.create({
                    data: {
                        mediaId,
                        width: metadata.width,
                        height: metadata.height,
                        format: metadata.format || null,
                        // ENABLED: These fields will now work correctly
                        fileSize: metadata.fileSize || null,
                        colorSpace: metadata.colorSpace || null,
                    },
                });
            case "AUDIO":
                return prisma.audioMetadata.create({
                    data: {
                        mediaId,
                        duration: metadata.duration,
                        bitrate: metadata.bitrate || null,
                        codec: metadata.codec || null,
                        channels: metadata.channels || null,
                    },
                });
            default:
                throw new Error(
                    `Invalid media type for metadata creation: ${mediaType}`
                );
        }
    },

    // UPDATED: Queries the 'media' table.
    async findById(mediaId) {
        return prisma.media.findUnique({
            where: { id: mediaId },
            ...mediaWithDetails,
        });
    },

    // UPDATED: Queries the 'media' table.
    async findByIdAndUserId(mediaId, userId) {
        return prisma.media.findFirst({
            where: { id: mediaId, userId },
            ...mediaWithDetails,
        });
    },

    // UPDATED: Queries the 'media' table.
    async findAllByUserId(userId) {
        return prisma.media.findMany({
            where: { userId },
            ...mediaWithDetails,
            orderBy: { createdAt: "desc" },
        });
    },

    // UPDATED: Queries the 'media' table.
    async updateStatus(mediaId, status) {
        return prisma.media.update({
            where: { id: mediaId },
            data: { status },
        });
    },

    async updateAnalysis(analysisId, data) {
        return prisma.deepfakeAnalysis.update({
            where: { id: analysisId },
            data,
        });
    },

    // UPDATED: Queries the 'media' table.
    async updateById(mediaId, updateData) {
        return prisma.media.update({
            where: { id: mediaId },
            data: updateData,
        });
    },

    // UPDATED: Queries the 'media' table.
    async deleteById(mediaId) {
        return prisma.media.delete({ where: { id: mediaId } });
    },

    // This function does not need changes as it only queries the DeepfakeAnalysis table.
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

    // --- REFACTORED: This is now the most powerful function in the repository. ---
    // It can intelligently save results for ANY media type by checking the shape of the resultData.
    async createAnalysisResult(mediaId, resultData) {
        const {
            // Common fields
            prediction,
            confidence,
            processingTime,
            model,
            modelVersion,
            analysisType,
            // Video-specific fields
            metrics,
            framePredictions,
            temporalAnalysis,
            // Audio-specific fields
            pitch,
            energy,
            spectral,
            visualization,
            // Common metadata
            modelInfo,
            systemInfo,
        } = resultData;

        return prisma.$transaction(async (tx) => {
            const analysis = await tx.deepfakeAnalysis.create({
                data: {
                    mediaId, // UPDATED
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

            // --- CONDITIONAL LOGIC FOR VIDEO DATA ---
            if (metrics) {
                await tx.analysisDetails.create({
                    data: {
                        analysisId: analysis.id,
                        frameCount:
                            metrics.frameCount ||
                            metrics.sequenceCount ||
                            metrics.totalFacesDetected ||
                            0,
                        avgConfidence:
                            metrics.finalAverageScore ||
                            metrics.averageFaceScore ||
                            0,
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

            // --- NEW: CONDITIONAL LOGIC FOR AUDIO DATA ---
            if (pitch && energy && spectral) {
                await tx.audioAnalysis.create({
                    data: {
                        analysisId: analysis.id,
                        meanPitchHz: pitch.mean_pitch_hz,
                        pitchStabilityScore: pitch.pitch_stability_score,
                        rmsEnergy: energy.rms_energy,
                        silenceRatio: energy.silence_ratio,
                        spectralCentroid: spectral.spectral_centroid,
                        spectralContrast: spectral.spectral_contrast,
                        spectrogramUrl: visualization?.spectrogram_url,
                        // Storing the raw data for potential client-side charting
                        spectrogramData:
                            visualization?.spectrogram_data || undefined,
                    },
                });
            }

            // --- COMMON METADATA ---
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

    // UPDATED: Changed videoId to mediaId.
    async createAnalysisError(mediaId, model, analysisType, error) {
        return prisma.$transaction(async (tx) => {
            const analysis = await tx.deepfakeAnalysis.create({
                data: {
                    mediaId, // UPDATED
                    model,
                    analysisType,
                    status: "FAILED",
                    errorMessage: error.message,
                    // These fields are non-nullable, so we provide defaults
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

    // These monitoring functions remain unchanged as they don't depend on the 'Video' model.
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
