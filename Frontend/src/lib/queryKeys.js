// src/lib/queryKeys.js

/**
 * A query key factory to ensure consistent and organized keys for TanStack Query.
 * This prevents typos and simplifies cache management.
 */
export const queryKeys = {
    // Authentication & User Profile
    auth: {
        all: ["auth"],
        profile: () => [...queryKeys.auth.all, "profile"],
    },

    // Video-related data
    videos: {
        all: ["videos"],
        lists: () => [...queryKeys.videos.all, "list"],
        detail: (id) => [...queryKeys.videos.all, "detail", id],
    },

    // Monitoring data from the backend
    monitoring: {
        all: ["monitoring"],
        serverStatus: () => [...queryKeys.monitoring.all, "serverStatus"],
        serverHistory: () => [...queryKeys.monitoring.all, "serverHistory"],
        analysisStats: () => [...queryKeys.monitoring.all, "analysisStats"],
        queueStatus: () => [...queryKeys.monitoring.all, "queueStatus"],
    },
};
