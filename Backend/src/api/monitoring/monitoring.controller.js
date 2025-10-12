// src/api/monitoring/monitoring.controller.js

import { mediaRepository } from '../../repositories/media.repository.js';
import { modelAnalysisService } from '../../services/modelAnalysis.service.js';
import { mediaQueue } from '../../config/index.js';
import { ApiResponse } from '../../utils/ApiResponse.js';
import { asyncHandler } from '../../utils/asyncHandler.js';
import { checkAndFinalizeStuckRuns } from '../../scripts/check-stuck-runs.js';
import { verifyAnalysisStatuses, fixIncorrectStatuses } from '../../scripts/verify-analysis-statuses.js';

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

const checkStuckRuns = asyncHandler(async (req, res) => {
    const result = await checkAndFinalizeStuckRuns();
    res.status(200).json(
        new ApiResponse(
            200, 
            result, 
            `Checked ${result.checked} runs and finalized ${result.finalized} stuck runs.`
        )
    );
});

const verifyStatuses = asyncHandler(async (req, res) => {
    const result = await verifyAnalysisStatuses();
    res.status(200).json(
        new ApiResponse(
            200, 
            result, 
            `Verified ${result.totalMedia} media items: ${result.correctMedia} correct, ${result.incorrectMedia} incorrect.`
        )
    );
});

const fixStatuses = asyncHandler(async (req, res) => {
    // First verify to get issues
    const verifyResult = await verifyAnalysisStatuses();
    
    if (verifyResult.incorrectMedia === 0) {
        return res.status(200).json(
            new ApiResponse(200, { fixed: 0, failed: 0 }, 'No status issues found.')
        );
    }
    
    // Fix the issues
    const fixResult = await fixIncorrectStatuses(verifyResult.issues);
    res.status(200).json(
        new ApiResponse(
            200, 
            fixResult, 
            `Fixed ${fixResult.fixed} status issues, ${fixResult.failed} failures.`
        )
    );
});

export const monitoringController = {
    getServerStatus,
    getServerHealthHistory,
    getQueueStatus,
    checkStuckRuns,
    verifyStatuses,
    fixStatuses,
};