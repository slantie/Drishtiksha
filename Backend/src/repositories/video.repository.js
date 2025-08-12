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

    async findAnalysis(videoId, model) {
        return prisma.deepfakeAnalysis.findUnique({
            where: {
                videoId_model_unique_constraint: {
                    videoId,
                    model,
                },
            },
        });
    },

    async createAnalysis(analysisData) {
        return prisma.deepfakeAnalysis.create({ data: analysisData });
    },
};
