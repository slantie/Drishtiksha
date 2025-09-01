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
        return prisma.deepfakeAnalysis.create({
            data: {
                analysisRunId: runId,
                modelName,
                prediction,
                confidence,
                status: 'COMPLETED',
                resultPayload,
            },
        });
    },

    async createAnalysisError(runId, modelName, error) {
        return prisma.deepfakeAnalysis.create({
            data: {
                analysisRunId: runId,
                modelName,
                status: 'FAILED',
                errorMessage: error.message,
                prediction: 'N/A',
                confidence: 0,
                resultPayload: { error: error.stack || error.message },
            },
        });
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
};