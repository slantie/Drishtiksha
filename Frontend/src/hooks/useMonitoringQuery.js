// src/hooks/useMonitoringQuery.js

import { useQuery } from "@tanstack/react-query";
import { monitoringApi } from "../services/api/monitoring.api.js";
import { queryKeys } from "../lib/queryKeys.js";

export const useServerStatusQuery = () => {
  return useQuery({
    queryKey: queryKeys.monitoring.serverStatus(),
    queryFn: monitoringApi.getServerStatus,
    refetchInterval: 30000, // 30 seconds
    select: (response) => response.data,
  });
};

export const useServerHistoryQuery = (params) => {
  return useQuery({
    queryKey: [...queryKeys.monitoring.serverHistory(), params],
    queryFn: () => monitoringApi.getServerHistory(params),
    select: (response) => response.data,
  });
};

export const useQueueStatusQuery = () => {
  return useQuery({
    queryKey: queryKeys.monitoring.queueStatus(),
    queryFn: monitoringApi.getQueueStatus,
    refetchInterval: 15000, // 15 seconds
    select: (response) => response.data,
  });
};
