// src/api/media/media.controller.js

import { mediaService } from '../../services/media.service.js';
import { asyncHandler } from '../../utils/asyncHandler.js';
import { ApiResponse } from '../../utils/ApiResponse.js';
import logger from '../../utils/logger.js';

const uploadMedia = asyncHandler(async (req, res) => {
    const { description } = req.body;
    const newMedia = await mediaService.createAndAnalyzeMedia(req.file, req.user, description);
    res.status(202).json(new ApiResponse(202, newMedia, 'Media uploaded and successfully queued for its first analysis run.'));
});

const rerunAnalysis = asyncHandler(async (req, res) => {
    const { id } = req.params;
    logger.info(`[API] Received request to re-run analysis for media ID: ${id}`);
    const updatedMedia = await mediaService.rerunAnalysis(id, req.user.id);
    res.status(202).json(new ApiResponse(202, updatedMedia, 'A new analysis run has been successfully queued for this media.'));
});

const getAllMedia = asyncHandler(async (req, res) => {
    const mediaItems = await mediaService.getAllMediaForUser(req.user.id);
    res.status(200).json(new ApiResponse(200, mediaItems));
});

const getMediaById = asyncHandler(async (req, res) => {
    const mediaItem = await mediaService.getMediaWithAnalyses(req.params.id, req.user.id);
    res.status(200).json(new ApiResponse(200, mediaItem));
});

const updateMedia = asyncHandler(async (req, res) => {
    const { id } = req.params;
    const updatedMedia = await mediaService.updateMedia(id, req.user.id, req.body);
    res.status(200).json(new ApiResponse(200, updatedMedia, 'Media updated successfully.'));
});

const deleteMedia = asyncHandler(async (req, res) => {
    await mediaService.deleteMediaById(req.params.id, req.user.id);
    res.status(200).json(new ApiResponse(200, {}, 'Media deleted successfully.'));
});

export const mediaController = {
    uploadMedia,
    rerunAnalysis,
    getAllMedia,
    getMediaById,
    updateMedia,
    deleteMedia,
};