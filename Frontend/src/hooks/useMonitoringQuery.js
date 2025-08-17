// src/hooks/useMonitoringQuery.js

import { useQuery } from "@tanstack/react-query";
import { monitoringApi } from "../services/api/monitoring.api.js";
import { queryKeys } from "../lib/queryKeys.js";

/**
 * Hook to fetch live status and statistics from the ML server.
 * Refetches every 30 seconds to provide near real-time data.
 */
export const useServerStatusQuery = () => {
    return useQuery({
        queryKey: queryKeys.monitoring.serverStatus(),
        queryFn: monitoringApi.getServerStatus,
        refetchInterval: 60000,
        select: (response) => response.data,
    });
};

/**
 * Hook to fetch the historical health data of the ML server.
 */
export const useServerHistoryQuery = (params) => {
    return useQuery({
        queryKey: [...queryKeys.monitoring.serverHistory(), params],
        queryFn: () => monitoringApi.getServerHistory(params),
        select: (response) => response.data,
    });
};

/**
 * Hook to fetch aggregated analysis statistics.
 */
export const useAnalysisStatsQuery = (params) => {
    return useQuery({
        queryKey: [...queryKeys.monitoring.analysisStats(), params],
        queryFn: () => monitoringApi.getAnalysisStats(params),
        select: (response) => response.data,
    });
};

/**
 * Hook to fetch the current status of the video processing queue.
 * Refetches every 15 seconds.
 */
export const useQueueStatusQuery = () => {
    return useQuery({
        queryKey: queryKeys.monitoring.queueStatus(),
        queryFn: monitoringApi.getQueueStatus,
        refetchInterval: 15000, // 15 seconds
        select: (response) => response.data,
    });
};
