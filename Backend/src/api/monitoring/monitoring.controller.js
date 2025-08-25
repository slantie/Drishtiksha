// src/api/monitoring/monitoring.controller.js

import { mediaRepository } from "../../repositories/media.repository.js";
import { modelAnalysisService } from "../../services/modelAnalysis.service.js";
// CORRECTED IMPORT: Points to the consolidated queue config file.
import { getQueueStatus } from "../../config/queue.js";
import { ApiResponse } from "../../utils/ApiResponse.js";
import { asyncHandler } from "../../utils/asyncHandler.js";
import logger from "../../utils/logger.js";
import { toCamelCase } from "../../utils/formatKeys.js";

const getServerStatus = asyncHandler(async (req, res) => {
    try {
        const serverStats = await modelAnalysisService.getServerStatistics();

        mediaRepository.storeServerHealth(serverStats).catch((err) => {
            logger.error(
                `Failed to store server health in background: ${err.message}`
            );
        });

        res.status(200).json(
            new ApiResponse(
                200,
                toCamelCase(serverStats),
                "Server status retrieved successfully"
            )
        );
    } catch (error) {
        mediaRepository
            .storeServerHealth({
                status: "UNHEALTHY",
                errorMessage: error.message,
            })
            .catch((err) => {
                logger.error(
                    `Failed to store FAILED server health in background: ${err.message}`
                );
            });

        throw error;
    }
});

const getServerHealthHistory = asyncHandler(async (req, res) => {
    const { limit = 50, serverUrl } = req.query;
    const history = await mediaRepository.getServerHealthHistory(
        serverUrl,
        parseInt(limit)
    );
    res.status(200).json(
        new ApiResponse(
            200,
            toCamelCase(history),
            "Server health history retrieved successfully"
        )
    );
});

const getAnalysisStats = asyncHandler(async (req, res) => {
    const { timeframe = "24h" } = req.query;
    const stats = await mediaRepository.getAnalysisStats(timeframe);
    res.status(200).json(
        new ApiResponse(
            200,
            toCamelCase(stats),
            `Analysis statistics for ${timeframe} retrieved successfully`
        )
    );
});

// CORRECTED: This now correctly calls the imported function without conflict.
const getQueueStatusHandler = asyncHandler(async (req, res) => {
    const queueStatus = await getQueueStatus();
    res.status(200).json(
        new ApiResponse(
            200,
            toCamelCase(queueStatus),
            "Video processing queue status retrieved successfully."
        )
    );
});

export const monitoringController = {
    getServerStatus,
    getServerHealthHistory,
    getAnalysisStats,
    getQueueStatus: getQueueStatusHandler,
};
