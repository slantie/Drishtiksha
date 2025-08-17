// src/constants/apiEndpoints.js

/**
 * @fileoverview Centralized API endpoints for the Drishtiksha application.
 * This file contains all the API endpoints used to communicate with the Node.js backend.
 */

// The base URL for the backend API, configured via environment variables.
export const API_BASE_URL =
    import.meta.env.VITE_BACKEND_URL || "http://localhost:3000";

export const API_ENDPOINTS = {
    // --- Authentication Routes ---
    AUTH: {
        LOGIN: `/auth/login`,
        SIGNUP: `/auth/signup`,
        LOGOUT: `/auth/logout`,
        PROFILE: `/auth/profile`,
        UPDATE_PASSWORD: `/auth/profile/password`,
        UPDATE_AVATAR: `/auth/profile/avatar`,
    },

    // --- Video Routes ---
    VIDEOS: {
        // GET all videos, POST to upload a new video
        ALL: `/videos`,
        // GET, PATCH, DELETE a specific video by its ID
        BY_ID: (id) => `/videos/${id}`,
    },

    // --- Monitoring Routes ---
    MONITORING: {
        // GET the live status of the ML server
        SERVER_STATUS: `/monitoring/server-status`,
        // GET historical health data for the ML server
        SERVER_HISTORY: `/monitoring/server-history`,
        // GET aggregated analysis performance statistics
        ANALYSIS_STATS: `/monitoring/stats/analysis`,
        // GET the status of the video processing queue
        QUEUE_STATUS: `/monitoring/queue-status`,
    },
};

// --- Constants for Analysis System ---

/** Supported analysis types as defined in the Prisma schema */
export const ANALYSIS_TYPES = {
    QUICK: "QUICK",
    DETAILED: "DETAILED",
    FRAMES: "FRAMES",
    VISUALIZE: "VISUALIZE",
    COMPREHENSIVE: "COMPREHENSIVE",
};

/** Supported model types (should match keys in `configs/config.yaml` on the server) */
export const MODEL_TYPES = {
    "SIGLIP-LSTM-V3": "SIGLIP-LSTM-V3",
    "COLOR-CUES-LSTM-V1": "COLOR-CUES-LSTM-V1",
};

/** Metadata for displaying analysis types in the UI */
export const ANALYSIS_TYPE_INFO = {
    [ANALYSIS_TYPES.COMPREHENSIVE]: {
        label: "Comprehensive Analysis",
        description:
            "Complete analysis with detailed metrics and confidence scores.",
        duration: "~1-3 minutes",
        icon: "🔍",
    },
    // Add other types here if they become individually selectable in the future
};

/** Metadata for displaying models in the UI */
export const MODEL_INFO = {
    [MODEL_TYPES["SIGLIP-LSTM-V3"]]: {
        label: "SigLIP LSTM v3",
        description: "Enhanced temporal analysis model with high accuracy.",
        version: "3.0.0",
        specialty: "General Purpose",
    },
    [MODEL_TYPES["COLOR-CUES-LSTM-V1"]]: {
        label: "Color Cues LSTM v1",
        description: "Specialized model focusing on color inconsistencies.",
        version: "1.0.0",
        specialty: "Color Artifacts",
    },
};
