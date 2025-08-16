// src/services/video.service.js

import { videoRepository } from "../repositories/video.repository.js";
// CORRECTED IMPORT: Points to the consolidated queue config file.
import { addVideoToQueue } from "../config/queue.js";
import {
    uploadOnCloudinary,
    deleteFromCloudinary,
} from "../utils/cloudinary.js";
import { ApiError } from "../utils/ApiError.js";
import logger from "../utils/logger.js";

class VideoService {
    async createVideoAndQueueForAnalysis(file, user, description, io) {
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

        // This now calls the function from the correct file.
        await addVideoToQueue(newVideo.id);

        this.emitVideoUpdate(io, newVideo.userId, newVideo);

        return newVideo;
    }

    emitVideoUpdate(io, userId, videoData) {
        if (io) {
            io.to(userId).emit("video_update", videoData);
            logger.info(
                `Emitted 'video_update' for video ${videoData.id} to user ${userId}.`
            );
        }
    }

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
