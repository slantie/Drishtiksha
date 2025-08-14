// src/api/videos/video.controller.js

import { videoService } from "../../services/video.service.js";
import { modelAnalysisService } from "../../services/modelAnalysis.service.js";
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

const getModelStatus = asyncHandler(async (req, res) => {
    const isAvailable = modelAnalysisService.isAvailable();
    let modelInfo = null;
    let healthStatus = null;

    if (isAvailable) {
        try {
            [healthStatus, modelInfo] = await Promise.allSettled([
                modelAnalysisService.checkHealth(),
                modelAnalysisService.getModelInfo(),
            ]);

            healthStatus =
                healthStatus.status === "fulfilled" ? healthStatus.value : null;
            modelInfo =
                modelInfo.status === "fulfilled" ? modelInfo.value : null;
        } catch (error) {
            // Health check failed, but that's okay - we'll report it
        }
    }

    res.status(200).json(
        new ApiResponse(
            200,
            {
                isConfigured: isAvailable,
                health: healthStatus,
                modelInfo: modelInfo,
            },
            "Model service status retrieved successfully."
        )
    );
});

const createVisualAnalysis = asyncHandler(async (req, res) => {
    const { id } = req.params;
    const updatedVideo = await videoService.createVisualAnalysis(id, req.user);
    res.status(200).json(
        new ApiResponse(
            200,
            updatedVideo,
            "Visual analysis generation started and completed successfully."
        )
    );
});

export const videoController = {
    uploadVideo,
    getAllVideos,
    getVideoById,
    updateVideo,
    deleteVideo,
    getModelStatus,
    createVisualAnalysis
};
