// src/services/video.service.js

import { videoRepository } from "../repositories/video.repository.js";
import { addAnalysisFlowToQueue } from "../config/queue.js";
import { modelAnalysisService } from "./modelAnalysis.service.js";
import {
    uploadOnCloudinary,
    deleteFromCloudinary,
} from "../utils/cloudinary.js";
import { ApiError } from "../utils/ApiError.js";
import logger from "../utils/logger.js";

class VideoService {
    // REMOVED: 'io' parameter is no longer needed.
        async createVideoAndQueueForAnalysis(file, user, description) {
        if (!file) throw new ApiError(400, "A video file is required.");

        const cloudinaryResponse = await uploadOnCloudinary(file.path);
        if (!cloudinaryResponse)
            throw new ApiError(500, "Failed to upload video.");

        const newVideo = await videoRepository.create({
            filename: file.originalname,
            description,
            url: cloudinaryResponse.secure_url,
            publicId: cloudinaryResponse.public_id,
            mimetype: file.mimetype,
            size: file.size,
            status: "QUEUED",
            userId: user.id,
        });

        // --- REFACTORED: Orchestration logic now lives here ---
        try {
            logger.info(`[VideoService] Creating analysis flow for video ${newVideo.id}`);
            const serverStats = await modelAnalysisService.getServerStatistics();
            const availableModels =
                serverStats.models_info?.filter((m) => m.loaded).map((m) => m.name) || [];

            if (availableModels.length === 0) {
                await videoRepository.updateStatus(newVideo.id, "FAILED");
                throw new ApiError(503, "No models are available for analysis.");
            }

            // Create a job for each available model
            const childJobs = availableModels.map((modelName) => ({
                name: "run-single-analysis",
                data: { videoId: newVideo.id, modelName, serverStats },
                queueName: "video-processing",
                opts: { jobId: `${newVideo.id}-${modelName}` },
            }));

            // Create the flow and add it to the queue
            await addAnalysisFlowToQueue(newVideo.id, childJobs);
            
            logger.info(`[VideoService] Successfully queued analysis flow for video ${newVideo.id} with ${childJobs.length} models.`);

        } catch (error) {
            logger.error(`[VideoService] Failed to create analysis flow for video ${newVideo.id}: ${error.message}`);
            // If flow creation fails, mark the video as failed
            await videoRepository.updateStatus(newVideo.id, "FAILED");
            throw new ApiError(500, "Could not queue video for analysis.", error.stack);
        }
        // --- END REFACTOR ---

        return newVideo;
    }

    // REMOVED: emitVideoUpdate method. This logic is now centralized.

    async getAllVideosForUser(userId) {
        return videoRepository.findAllByUserId(userId);
    }

    async getVideoWithAnalyses(videoId, userId) {
        const video = await videoRepository.findByIdAndUserId(videoId, userId);
        if (!video)
            throw new ApiError(
                404,
                "Video not found or you do not have permission."
            );
        return video;
    }

    // Route to Update basic Video details (only filename and description)
    async updateVideo(videoId, userId, updateData) {
        const video = await this.getVideoWithAnalyses(videoId, userId);
        if (!video) throw new ApiError(404, "Video not found.");

        // Update only the allowed fields
        const allowedFields = ["filename", "description"];
        for (const key of Object.keys(updateData)) {
            if (!allowedFields.includes(key)) {
                throw new ApiError(400, `Field '${key}' is not updatable.`);
            }
        }

        const updatedVideo = await videoRepository.updateById(videoId, updateData);
        return updatedVideo;
    }

    async deleteVideoById(videoId, userId) {
        const video = await this.getVideoWithAnalyses(videoId, userId);
        if (video.publicId) await deleteFromCloudinary(video.publicId, "video");

        for (const analysis of video.analyses) {
            if (analysis.visualizedUrl) {
                try {
                    const publicId = analysis.visualizedUrl
                        .split("/")
                        .pop()
                        .split(".")[0];
                    await deleteFromCloudinary(
                        `deepfake-visualizations/${publicId}`,
                        "video"
                    );
                } catch (e) {
                    logger.warn(
                        `Could not delete visualization from Cloudinary: ${e.message}`
                    );
                }
            }
        }

        await videoRepository.deleteById(videoId);
    }
}

export const videoService = new VideoService();
