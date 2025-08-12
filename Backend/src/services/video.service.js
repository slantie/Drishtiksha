// src/services/video.service.js

import { videoRepository } from "../repositories/video.repository.js";
import {
    uploadOnCloudinary,
    deleteFromCloudinary,
} from "../utils/cloudinary.js";
import { ApiError } from "../utils/ApiError.js";
import { videoProcessorQueue } from "../queue/videoProcessorQueue.js";
import logger from "../utils/logger.js";

const ANALYSIS_MODELS = ["SIGLIPV1", "RPPG", "COLORCUES"];

const generateMockAnalysis = (filename, fileSize, model) => {
    const isLikelyReal = Math.random() > 0.3;
    const baseConfidence = isLikelyReal
        ? 0.75 + Math.random() * 0.2
        : 0.6 + Math.random() * 0.3;
    const confidence = parseFloat(
        Math.min(0.98, Math.max(0.52, baseConfidence)).toFixed(4)
    );
    const prediction = confidence > 0.75 ? "REAL" : "FAKE";
    const processingTime = parseFloat(
        (
            2.5 +
            (fileSize / (1024 * 1024)) * 0.1 +
            (Math.random() - 0.5)
        ).toFixed(2)
    );
    return {
        prediction,
        confidence,
        processingTime,
        model,
        status: "COMPLETED",
    };
};

export const videoService = {
    async uploadAndProcessVideo(file, description, user) {
        if (!file) {
            throw new ApiError(400, "No video file provided");
        }

        const cloudinaryResponse = await uploadOnCloudinary(file.path);
        if (!cloudinaryResponse) {
            throw new ApiError(500, "Video failed to upload on Cloudinary");
        }

        const newVideo = await videoRepository.create({
            filename: file.originalname,
            mimetype: file.mimetype,
            size: file.size,
            description: description || "",
            url: cloudinaryResponse.secure_url,
            publicId: cloudinaryResponse.public_id,
            userId: user.id,
            status: "UPLOADED",
        });

        // Add job to the queue for automatic multi-model analysis
        videoProcessorQueue({ videoId: newVideo.id, userId: newVideo.userId });
        logger.info(
            `Video ${newVideo.id} added to in-memory processing queue.`
        );

        return newVideo;
    },

    async getAllVideosForUser(user) {
        if (user.role === "ADMIN") {
            return videoRepository.findAll();
        }
        return videoRepository.findAllByUser(user.id);
    },

    async getVideoById(videoId, user) {
        const video = await videoRepository.findById(videoId);
        if (!video) {
            throw new ApiError(404, "Video not found");
        }
        if (video.userId !== user.id && user.role !== "ADMIN") {
            throw new ApiError(
                403,
                "Access denied. You do not own this video."
            );
        }
        return video;
    },

    async updateVideoDetails(videoId, updateData, user) {
        const video = await this.getVideoById(videoId, user); // Reuse permission check
        return videoRepository.update(videoId, updateData);
    },

    async deleteVideoById(videoId, user) {
        const video = await this.getVideoById(videoId, user); // Reuse permission check

        if (video.publicId) {
            await deleteFromCloudinary(video.publicId, "video");
        }

        await videoRepository.delete(videoId);
    },

    async runFullAnalysis(videoId) {
        logger.info(`Starting multi-model analysis for video ID: ${videoId}`);
        const video = await videoRepository.findById(videoId);
        if (!video) {
            logger.error(`Video with ID ${videoId} not found for analysis.`);
            return;
        }

        await videoRepository.update(videoId, { status: "PROCESSING" });

        try {
            for (const model of ANALYSIS_MODELS) {
                const existing = video.analyses.find((a) => a.model === model);
                if (existing) {
                    logger.info(
                        `Analysis for model ${model} already exists for video ${videoId}. Skipping.`
                    );
                    continue;
                }

                logger.info(
                    `Running mock analysis for video ${videoId} with model ${model}.`
                );
                await new Promise((resolve) =>
                    setTimeout(resolve, 1000 + Math.random() * 1500)
                );
                const results = generateMockAnalysis(
                    video.filename,
                    video.size,
                    model
                );
                await videoRepository.createAnalysis({ videoId, ...results });
            }
            await videoRepository.update(videoId, { status: "ANALYZED" });
            logger.info(`All analyses for video ${videoId} are now complete.`);
        } catch (error) {
            logger.error(
                `Failed to complete video analysis for ID ${videoId}: ${error.message}`
            );
            await videoRepository.update(videoId, { status: "FAILED" });
        }
    },
};
