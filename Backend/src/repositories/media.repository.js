// src/repositories/media.repository.js

import { prisma } from '../config/index.js';

const mediaWithLatestRunDetails = {
    include: {
        user: {
            select: { id: true, firstName: true, lastName: true, email: true },
        },
        analysisRuns: {
            orderBy: { runNumber: 'desc' },
            take: 1,
            include: {
                analyses: {
                    orderBy: { createdAt: 'asc' },
                },
            },
        },
    },
};

const mediaWithAllRunsDetails = {
    include: {
        user: {
            select: { id: true, firstName: true, lastName: true, email: true },
        },
        analysisRuns: {
            orderBy: { runNumber: 'desc' },
            include: {
                analyses: {
                    orderBy: { createdAt: 'asc' },
                },
            },
        },
    },
};

export const mediaRepository = {
    // --- Media Operations ---
    async create(mediaData) {
        return prisma.media.create({ data: mediaData });
    },

    async findById(mediaId) {
        return prisma.media.findUnique({
            where: { id: mediaId },
            ...mediaWithAllRunsDetails,
        });
    },

    async findByIdAndUserId(mediaId, userId) {
        return prisma.media.findFirst({
            where: { id: mediaId, userId },
            ...mediaWithAllRunsDetails,
        });
    },

    async findAllByUserId(userId) {
        return prisma.media.findMany({
            where: { userId },
            ...mediaWithLatestRunDetails,
            orderBy: { createdAt: 'desc' },
        });
    },

    async update(mediaId, updateData) {
        return prisma.media.update({
            where: { id: mediaId },
            data: updateData,
        });
    },

    async deleteById(mediaId) {
        return prisma.media.delete({ where: { id: mediaId } });
    },

    // --- Analysis Run Operations ---
    async createAnalysisRun(mediaId, runNumber) {
        return prisma.analysisRun.create({
            data: { mediaId, runNumber, status: 'QUEUED' },
        });
    },

    async updateRunStatus(runId, status) {
        return prisma.analysisRun.update({
            where: { id: runId },
            data: { status },
        });
    },

    async findLatestRunNumber(mediaId) {
        const latestRun = await prisma.analysisRun.findFirst({
            where: { mediaId },
            orderBy: { runNumber: 'desc' },
            select: { runNumber: true },
        });
        return latestRun?.runNumber || 0;
    },
    
    // --- Analysis Result Operations ---
    async createAnalysisResult(runId, resultData) {
        const { modelName, prediction, confidence, resultPayload } = resultData;
        
        // ✨ Extract promoted fields from resultPayload for efficient querying
        const processingTime = resultPayload?.processing_time || resultPayload?.processingTime || null;
        const mediaType = resultPayload?.media_type || resultPayload?.mediaType || null;
        
        return prisma.deepfakeAnalysis.create({
            data: {
                analysisRunId: runId,
                modelName,
                prediction,
                confidence,
                status: 'COMPLETED',
                processingTime,  // ✨ NEW: Promoted field
                mediaType,       // ✨ NEW: Promoted field
                resultPayload,
            },
        });
    },

    async createAnalysisError(runId, modelName, error) {
        // Ensure errorMessage is always a string
        let errorMessage;
        if (typeof error.message === 'string') {
            errorMessage = error.message;
        } else if (typeof error.message === 'object') {
            // If message is an object, stringify it
            errorMessage = JSON.stringify(error.message);
        } else {
            errorMessage = String(error.message || error);
        }
        
        // 🔍 BUGFIX: Check if an analysis entry already exists for this run + model combo
        // This prevents duplicate FAILED entries when BullMQ retries the job
        const existingAnalysis = await prisma.deepfakeAnalysis.findFirst({
            where: {
                analysisRunId: runId,
                modelName: modelName,
            },
        });
        
        // Build result payload with full error details
        const resultPayload = {
            error: errorMessage,
            stack: error.stack,
            // Include server response if available (from ApiError)
            serverResponse: error.serverResponse || error.details,
            timestamp: new Date().toISOString(),
            // Track retry count to show users how many times the job was retried
            retryCount: existingAnalysis ? ((existingAnalysis.resultPayload?.retryCount || 0) + 1) : 0,
        };
        
        if (existingAnalysis) {
            // 🔄 UPDATE: If entry exists, update it with new error details and incremented retry count
            return prisma.deepfakeAnalysis.update({
                where: { id: existingAnalysis.id },
                data: {
                    status: 'FAILED',
                    errorMessage: errorMessage, // Guaranteed to be string
                    resultPayload: resultPayload,
                },
            });
        } else {
            // ✨ CREATE: If this is the first failure, create a new entry
            return prisma.deepfakeAnalysis.create({
                data: {
                    analysisRunId: runId,
                    modelName,
                    status: 'FAILED',
                    errorMessage: errorMessage, // Guaranteed to be string
                    prediction: 'N/A',
                    confidence: 0,
                    resultPayload: resultPayload,
                },
            });
        }
    },

    // --- Monitoring Operations ---
    async storeServerHealth(healthData) {
        const mapStatus = (s) => {
            const statusStr = String(s?.status).toUpperCase();
            if (['OK', 'HEALTHY', 'RUNNING'].includes(statusStr)) return 'HEALTHY';
            if (['DEGRADED'].includes(statusStr)) return 'DEGRADED';
            if (['UNHEALTHY', 'ERROR', 'FAILED'].includes(statusStr)) return 'UNHEALTHY';
            return 'UNKNOWN';
        };

        return prisma.serverHealth.create({
            data: {
                status: mapStatus(healthData),
                responseTimeMs: healthData.responseTimeMs || 0,
                statsPayload: healthData || {},
                checkedAt: new Date(),
            },
        });
    },

    async getServerHealthHistory(limit = 50) {
        return prisma.serverHealth.findMany({
            orderBy: { checkedAt: 'desc' },
            take: limit,
        });
    },

    // --- Analytics & Query Operations (Using Promoted Fields) ---
    
    /**
     * Get analyses filtered by processing time range.
     * Enables queries like "show all analyses that took less than 10 seconds"
     */
    async getAnalysesByProcessingTime(minTime, maxTime, options = {}) {
        const { limit = 100, includeRelations = false } = options;
        
        return prisma.deepfakeAnalysis.findMany({
            where: {
                processingTime: {
                    gte: minTime,
                    lte: maxTime,
                },
                status: 'COMPLETED', // Only completed analyses
            },
            ...(includeRelations && {
                include: {
                    analysisRun: {
                        include: { media: true },
                    },
                },
            }),
            orderBy: { processingTime: 'asc' },
            take: limit,
        });
    },

    /**
     * Get analyses filtered by media type.
     * Enables queries like "show all video analyses" or "show all image analyses"
     */
    async getAnalysesByMediaType(mediaType, options = {}) {
        const { limit = 100, includeRelations = false } = options;
        
        return prisma.deepfakeAnalysis.findMany({
            where: { 
                mediaType,
                status: 'COMPLETED',
            },
            ...(includeRelations && {
                include: {
                    analysisRun: {
                        include: { media: true },
                    },
                },
            }),
            orderBy: { createdAt: 'desc' },
            take: limit,
        });
    },

    /**
     * Get average processing time per model.
     * Useful for performance monitoring and model comparison.
     */
    async getAverageProcessingTimeByModel() {
        return prisma.$queryRaw`
            SELECT 
                model_name as "modelName",
                COUNT(*) as "totalAnalyses",
                AVG(processing_time) as "avgProcessingTime",
                MIN(processing_time) as "minProcessingTime",
                MAX(processing_time) as "maxProcessingTime"
            FROM deepfake_analyses
            WHERE processing_time IS NOT NULL 
            AND status = 'COMPLETED'
            GROUP BY model_name
            ORDER BY "avgProcessingTime" ASC
        `;
    },

    /**
     * Get average confidence by model and media type.
     * Enables insights like "which model performs best on videos?"
     */
    async getAverageConfidenceByModelAndMediaType() {
        return prisma.$queryRaw`
            SELECT 
                model_name as "modelName",
                media_type as "mediaType",
                COUNT(*) as "totalAnalyses",
                AVG(confidence) as "avgConfidence",
                MIN(confidence) as "minConfidence",
                MAX(confidence) as "maxConfidence"
            FROM deepfake_analyses
            WHERE media_type IS NOT NULL 
            AND status = 'COMPLETED'
            GROUP BY model_name, media_type
            ORDER BY "avgConfidence" DESC
        `;
    },

    /**
     * Get slowest analyses for debugging/optimization.
     */
    async getSlowestAnalyses(limit = 10) {
        return prisma.deepfakeAnalysis.findMany({
            where: {
                processingTime: { not: null },
                status: 'COMPLETED',
            },
            include: {
                analysisRun: {
                    include: { 
                        media: {
                            select: { id: true, filename: true, mediaType: true }
                        }
                    },
                },
            },
            orderBy: { processingTime: 'desc' },
            take: limit,
        });
    },

    /**
     * Get fastest analyses for performance benchmarking.
     */
    async getFastestAnalyses(limit = 10) {
        return prisma.deepfakeAnalysis.findMany({
            where: {
                processingTime: { not: null },
                status: 'COMPLETED',
            },
            include: {
                analysisRun: {
                    include: { 
                        media: {
                            select: { id: true, filename: true, mediaType: true }
                        }
                    },
                },
            },
            orderBy: { processingTime: 'asc' },
            take: limit,
        });
    },
};