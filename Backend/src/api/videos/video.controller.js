// src/api/videos/video.controller.js

import { videoService } from "../../services/video.service.js";
import { asyncHandler } from "../../utils/asyncHandler.js";
import { ApiResponse } from "../../utils/ApiResponse.js";

const uploadVideo = asyncHandler(async (req, res) => {
    const { description } = req.body;
    const newVideo = await videoService.uploadAndProcessVideo(
        req.file,
        description,
        req.user
    );
    res.status(201).json(
        new ApiResponse(
            201,
            newVideo,
            "Video uploaded successfully and is now processing."
        )
    );
});

const getAllVideos = asyncHandler(async (req, res) => {
    const videos = await videoService.getAllVideosForUser(req.user);
    res.status(200).json(
        new ApiResponse(200, videos, "Videos retrieved successfully.")
    );
});

const getVideoById = asyncHandler(async (req, res) => {
    const video = await videoService.getVideoById(req.params.id, req.user);
    res.status(200).json(
        new ApiResponse(200, video, "Video retrieved successfully.")
    );
});

const updateVideo = asyncHandler(async (req, res) => {
    const updatedVideo = await videoService.updateVideoDetails(
        req.params.id,
        req.body,
        req.user
    );
    res.status(200).json(
        new ApiResponse(200, updatedVideo, "Video Data updated successfully.")
    );
});

const deleteVideo = asyncHandler(async (req, res) => {
    await videoService.deleteVideoById(req.params.id, req.user);
    res.status(200).json(
        new ApiResponse(200, null, "Video deleted successfully.")
    );
});

export const videoController = {
    uploadVideo,
    getAllVideos,
    getVideoById,
    updateVideo,
    deleteVideo,
};
