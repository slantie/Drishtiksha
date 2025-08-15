// src/repositories/video.repository.js

import prisma from "../config/database.js";

const videoWithDetails = {
    user: {
        select: { id: true, firstName: true, lastName: true, email: true },
    },
    analyses: { orderBy: { createdAt: "desc" } },
};

export const videoRepository = {
    async create(videoData) {
        return prisma.video.create({ data: videoData });
    },

    async findById(videoId) {
        return prisma.video.findUnique({
            where: { id: videoId },
            include: videoWithDetails,
        });
    },

    async findAllByUser(userId) {
        return prisma.video.findMany({
            where: { userId },
            include: videoWithDetails,
            orderBy: { createdAt: "desc" },
        });
    },

    async findAll() {
        return prisma.video.findMany({
            include: videoWithDetails,
            orderBy: { createdAt: "desc" },
        });
    },

    async update(videoId, updateData) {
        return prisma.video.update({
            where: { id: videoId },
            data: updateData,
            include: videoWithDetails,
        });
    },

    async delete(videoId) {
        return prisma.video.delete({ where: { id: videoId } });
    },

    async findAnalysis(videoId, model, analysisType = null) {
        if (analysisType) {
            // Find the most recent analysis for this combination
            return prisma.deepfakeAnalysis.findFirst({
                where: {
                    videoId,
                    model,
                    analysisType,
                },
                include: {
                    analysisDetails: true,
                    frameAnalysis: true,
                    temporalAnalysis: true,
                    modelInfo: true,
                    systemInfo: true,
                    errors: true,
                },
                orderBy: { createdAt: "desc" },
            });
        }

        // For backward compatibility, find any analysis for video+model
        return prisma.deepfakeAnalysis.findFirst({
            where: {
                videoId,
                model,
            },
            include: {
                analysisDetails: true,
                frameAnalysis: true,
                temporalAnalysis: true,
                modelInfo: true,
                systemInfo: true,
                errors: true,
            },
            orderBy: { createdAt: "desc" },
        });
    },

    async createAnalysis(analysisData) {
        return prisma.deepfakeAnalysis.create({
            data: analysisData,
            include: {
                analysisDetails: true,
                frameAnalysis: true,
                temporalAnalysis: true,
                modelInfo: true,
                systemInfo: true,
                errors: true,
            },
        });
    },

    async updateAnalysis(analysisId, updateData) {
        return prisma.deepfakeAnalysis.update({
            where: { id: analysisId },
            data: updateData,
            include: {
                analysisDetails: true,
                frameAnalysis: true,
                temporalAnalysis: true,
                modelInfo: true,
                systemInfo: true,
                errors: true,
            },
        });
    },

    async findAnalysesByVideo(videoId) {
        return prisma.deepfakeAnalysis.findMany({
            where: { videoId },
            include: {
                analysisDetails: true,
                frameAnalysis: true,
                temporalAnalysis: true,
                modelInfo: true,
                systemInfo: true,
                errors: true,
            },
            orderBy: { createdAt: "desc" },
        });
    },

    async findAnalysesByType(videoId, analysisType) {
        return prisma.deepfakeAnalysis.findMany({
            where: {
                videoId,
                analysisType,
            },
            include: {
                analysisDetails: true,
                frameAnalysis: true,
                temporalAnalysis: true,
                modelInfo: true,
                systemInfo: true,
                errors: true,
            },
            orderBy: { createdAt: "desc" },
        });
    },
};
