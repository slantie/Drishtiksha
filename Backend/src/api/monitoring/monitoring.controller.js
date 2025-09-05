// src/api/monitoring/monitoring.controller.js

import { mediaRepository } from '../../repositories/media.repository.js';
import { modelAnalysisService } from '../../services/modelAnalysis.service.js';
import { mediaQueue } from '../../config/index.js';
import { ApiResponse } from '../../utils/ApiResponse.js';
import { asyncHandler } from '../../utils/asyncHandler.js';

const getServerStatus = asyncHandler(async (req, res) => {
    const serverStats = await modelAnalysisService.getServerStatistics();
    res.status(200).json(new ApiResponse(200, serverStats, 'Server status retrieved successfully'));
});

const getServerHealthHistory = asyncHandler(async (req, res) => {
    const limit = req.query.limit ? parseInt(req.query.limit, 10) : 50;
    const history = await mediaRepository.getServerHealthHistory(limit);
    res.status(200).json(new ApiResponse(200, history, 'Server health history retrieved successfully'));
});

const getQueueStatus = asyncHandler(async (req, res) => {
    const status = {
        pending: await mediaQueue.getWaitingCount(),
        active: await mediaQueue.getActiveCount(),
        completed: await mediaQueue.getCompletedCount(),
        failed: await mediaQueue.getFailedCount(),
        delayed: await mediaQueue.getDelayedCount(),
    };
    res.status(200).json(new ApiResponse(200, status, 'Processing queue status retrieved successfully.'));
});

export const monitoringController = {
    getServerStatus,
    getServerHealthHistory,
    getQueueStatus,
};