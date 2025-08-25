// src/api/media/media.controller.js

// RENAMED: Importing mediaService
import { mediaService } from "../../services/media.service.js";
import { asyncHandler } from "../../utils/asyncHandler.js";
import { ApiResponse } from "../../utils/ApiResponse.js";

// RENAMED: from uploadVideo to uploadMedia
const uploadMedia = asyncHandler(async (req, res) => {
    const { description } = req.body;

    const newMedia = await mediaService.createMediaAndQueueForAnalysis(
        req.file,
        req.user,
        description
    );

    res.status(202).json(
        new ApiResponse(
            202,
            newMedia,
            // UPDATED: Generic success message
            "Media uploaded and queued for analysis."
        )
    );
});

// RENAMED: from getAllVideos to getAllMedia
const getAllMedia = asyncHandler(async (req, res) => {
    const mediaItems = await mediaService.getAllMediaForUser(req.user.id);
    res.status(200).json(new ApiResponse(200, mediaItems));
});

// RENAMED: from getVideoById to getMediaById
const getMediaById = asyncHandler(async (req, res) => {
    const mediaItem = await mediaService.getMediaWithAnalyses(
        req.params.id,
        req.user.id
    );
    res.status(200).json(new ApiResponse(200, mediaItem));
});

// RENAMED: from updateVideo to updateMedia
const updateMedia = asyncHandler(async (req, res) => {
    const { id } = req.params;
    const updateData = req.body;

    const updatedMedia = await mediaService.updateMedia(
        id,
        req.user.id,
        updateData
    );

    res.status(200).json(
        new ApiResponse(200, updatedMedia, "Media updated successfully.")
    );
});

// RENAMED: from deleteVideo to deleteMedia
const deleteMedia = asyncHandler(async (req, res) => {
    await mediaService.deleteMediaById(req.params.id, req.user.id);
    res.status(200).json(
        new ApiResponse(200, {}, "Media deleted successfully.")
    );
});

// RENAMED: Exporting mediaController
export const mediaController = {
    uploadMedia,
    getAllMedia,
    getMediaById,
    updateMedia,
    deleteMedia,
};
