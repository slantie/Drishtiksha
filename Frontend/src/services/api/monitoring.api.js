// src/services/api/monitoring.api.js

import axiosInstance from "../../lib/axios.js";

const MONITORING_ROUTES = {
  SERVER_STATUS: "/monitoring/server-status",
  SERVER_HISTORY: "/monitoring/server-history",
  QUEUE_STATUS: "/monitoring/queue-status",
};

export const monitoringApi = {
  getServerStatus: async () =>
    axiosInstance.get(MONITORING_ROUTES.SERVER_STATUS),
  getServerHistory: async (params) =>
    axiosInstance.get(MONITORING_ROUTES.SERVER_HISTORY, { params }),
  getQueueStatus: async () => axiosInstance.get(MONITORING_ROUTES.QUEUE_STATUS),
};
