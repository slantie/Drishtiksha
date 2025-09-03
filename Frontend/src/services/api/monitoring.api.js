// src/services/api/monitoring.api.js

import axiosInstance from "../../lib/axios.js";

const MONITORING_ROUTES = {
  SERVER_STATUS: "/api/v1/monitoring/server-status", // Explicitly include API versioning
  SERVER_HISTORY: "/api/v1/monitoring/server-history", // Explicitly include API versioning
  QUEUE_STATUS: "/api/v1/monitoring/queue-status", // Explicitly include API versioning
};

export const monitoringApi = {
  getServerStatus: async () =>
    axiosInstance.get(MONITORING_ROUTES.SERVER_STATUS),
  getServerHistory: async (params) =>
    axiosInstance.get(MONITORING_ROUTES.SERVER_HISTORY, { params }),
  getQueueStatus: async () => axiosInstance.get(MONITORING_ROUTES.QUEUE_STATUS),
};
