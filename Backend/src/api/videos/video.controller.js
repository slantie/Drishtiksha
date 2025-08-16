// src/api/videos/video.controller.js

import { videoService } from "../../services/video.service.js";
import { modelAnalysisService } from "../../services/modelAnalysis.service.js";
import { asyncHandler } from "../../utils/asyncHandler.js";
import { ApiResponse } from "../../utils/ApiResponse.js";

const uploadVideo = asyncHandler(async (req, res) => {
    const { description } = req.body;
    // ADDED: Get the io instance from the request object.
    const io = req.app.get("io");

    // CHANGED: Pass the io instance to the service layer.
    const newVideo = await videoService.createVideoAndQueueForAnalysis(
        req.file,
        req.user,
        description,
        io // Pass the io instance
    );

    res.status(202).json(
        new ApiResponse(
            202,
            newVideo,
            "Video uploaded and queued for analysis."
        )
    );
});

const getAllVideos = asyncHandler(async (req, res) => {
    const videos = await videoService.getAllVideosForUser(req.user.id);
    res.status(200).json(
        new ApiResponse(200, videos, "Videos retrieved successfully.")
    );
});

const getVideoById = asyncHandler(async (req, res) => {
    const video = await videoService.getVideoWithAnalyses(
        req.params.id,
        req.user.id
    );
    res.status(200).json(
        new ApiResponse(200, video, "Video retrieved successfully.")
    );
});

const deleteVideo = asyncHandler(async (req, res) => {
    await videoService.deleteVideoById(req.params.id, req.user.id);
    res.status(200).json(
        new ApiResponse(200, {}, "Video deleted successfully.")
    );
});

export const videoController = {
    uploadVideo,
    getAllVideos,
    getVideoById,
    deleteVideo,
};
