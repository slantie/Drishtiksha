// src/services/api/monitoring.api.js

import axiosInstance from "../../lib/axios.js";
import { API_ENDPOINTS } from "../../constants/apiEndpoints.js";

export const monitoringApi = {
    /**
     * Fetches the live, comprehensive statistics from the ML server.
     * Includes device info, system info, model status, etc.
     * @returns {Promise<object>} The API response containing server statistics.
     */
    getServerStatus: async () => {
        return await axiosInstance.get(API_ENDPOINTS.MONITORING.SERVER_STATUS);
    },

    /**
     * Fetches the historical health check data for the ML server.
     * @param {object} params - Optional query parameters { limit, serverUrl }.
     * @returns {Promise<object>} The API response containing an array of health records.
     */
    getServerHistory: async (params) => {
        return await axiosInstance.get(
            API_ENDPOINTS.MONITORING.SERVER_HISTORY,
            { params }
        );
    },

    /**
     * Fetches aggregated analysis statistics over a given timeframe.
     * @param {object} params - Optional query parameters { timeframe }.
     * @returns {Promise<object>} The API response containing aggregated stats.
     */
    getAnalysisStats: async (params) => {
        return await axiosInstance.get(
            API_ENDPOINTS.MONITORING.ANALYSIS_STATS,
            { params }
        );
    },

    /**
     * Fetches the current status of the BullMQ video processing queue.
     * @returns {Promise<object>} The API response containing queue job counts.
     */
    getQueueStatus: async () => {
        return await axiosInstance.get(API_ENDPOINTS.MONITORING.QUEUE_STATUS);
    },
};
