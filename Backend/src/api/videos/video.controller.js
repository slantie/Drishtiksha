// src/api/videos/video.controller.js

import { videoService } from "../../services/video.service.js";
import { modelAnalysisService } from "../../services/modelAnalysis.service.js";
import { videoRepository } from "../../repositories/video.repository.js";
import { asyncHandler } from "../../utils/asyncHandler.js";
import { ApiResponse } from "../../utils/ApiResponse.js";
import { ApiError } from "../../utils/ApiError.js";

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

const createVisualAnalysis = asyncHandler(async (req, res) => {
    const { id } = req.params;
    const { model } = req.body; // Optional specific model

    const updatedVideo = await videoService.createVisualAnalysis(
        id,
        req.user,
        model
    );
    res.status(200).json(
        new ApiResponse(
            200,
            updatedVideo,
            "Visual analysis generation started and completed successfully."
        )
    );
});

const createSpecificAnalysis = asyncHandler(async (req, res) => {
    const { id } = req.params;
    const { type, model } = req.body;

    if (!type) {
        throw new ApiError(400, "Analysis type is required");
    }

    const results = await videoService.createSpecificAnalysis(
        id,
        req.user,
        type,
        model
    );
    res.status(200).json(
        new ApiResponse(
            200,
            results,
            `${type} analysis completed successfully.`
        )
    );
});

const getAnalysisResults = asyncHandler(async (req, res) => {
    const { id } = req.params;
    let { type, model } = req.query;

    // Handle string "undefined" values from query parameters
    if (type === "undefined" || type === "") type = undefined;
    if (model === "undefined" || model === "") model = undefined;

    const video = await videoService.getVideoById(id, req.user);

    let analyses;
    if (type && model) {
        // Get specific analysis
        analyses = await videoRepository.findAnalysis(video.id, model, type);
    } else if (type) {
        // Get all analyses of a specific type
        analyses = await videoRepository.findAnalysesByType(video.id, type);
    } else {
        // Get all analyses for the video
        analyses = await videoRepository.findAnalysesByVideo(video.id);
    }

    res.status(200).json(
        new ApiResponse(
            200,
            analyses,
            "Analysis results retrieved successfully."
        )
    );
});

const getModelStatus = asyncHandler(async (req, res) => {
    const isAvailable = modelAnalysisService.isAvailable();
    let modelInfo = null;
    let healthStatus = null;
    let availableModels = [];

    if (isAvailable) {
        try {
            [healthStatus, modelInfo, availableModels] =
                await Promise.allSettled([
                    modelAnalysisService.checkHealth(),
                    modelAnalysisService.getModelInfo(),
                    modelAnalysisService.getAvailableModels(),
                ]);

            healthStatus =
                healthStatus.status === "fulfilled" ? healthStatus.value : null;
            modelInfo =
                modelInfo.status === "fulfilled" ? modelInfo.value : null;
            availableModels =
                availableModels.status === "fulfilled"
                    ? availableModels.value
                    : [];
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
                availableModels: availableModels,
                supportedAnalysisTypes: [
                    "QUICK",
                    "DETAILED",
                    "FRAMES",
                    "VISUALIZE",
                ],
            },
            "Model service status retrieved successfully."
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
    createVisualAnalysis,
    createSpecificAnalysis,
    getAnalysisResults,
};
