// src/constants/apiEndpoints.js

/**
 * @fileoverview Centralized API endpoints for the Drishtiksha application.
 * This file contains all the API endpoints used to communicate with the Node.js backend.
 */

// The base URL for the backend API, configured via environment variables.
export const API_BASE_URL =
    import.meta.env.VITE_BACKEND_URL || "http://localhost:3000";

export const API_ENDPOINTS = {
    // --- Authentication Routes (No Changes) ---
    AUTH: {
        LOGIN: `/auth/login`,
        SIGNUP: `/auth/signup`,
        LOGOUT: `/auth/logout`,
        PROFILE: `/auth/profile`,
        UPDATE_PASSWORD: `/auth/profile/password`,
        UPDATE_AVATAR: `/auth/profile/avatar`,
    },

    // --- REFACTORED: From VIDEOS to MEDIA ---
    MEDIA: {
        // GET all media, POST to upload new media
        ALL: `/media`,
        // GET, PATCH, DELETE a specific media item by its ID
        BY_ID: (id) => `/media/${id}`,
    },

    // --- Monitoring Routes (No Changes) ---
    MONITORING: {
        SERVER_STATUS: `/monitoring/server-status`,
        SERVER_HISTORY: `/monitoring/server-history`,
        ANALYSIS_STATS: `/monitoring/stats/analysis`,
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

/**
 * UPDATED: Model types now reflect the full suite from the Python server.
 * This ensures the UI can display correct labels and descriptions.
 */
export const MODEL_TYPES = {
    "SIGLIP-LSTM-V4": "SIGLIP-LSTM-V4",
    "EFFICIENTNET-B7-V1": "EFFICIENTNET-B7-V1",
    "COLOR-CUES-LSTM-V1": "COLOR-CUES-LSTM-V1",
    "EYEBLINK-CNN-LSTM-V1": "EYEBLINK-CNN-LSTM-V1",
    "SCATTERING-WAVE-V1": "SCATTERING-WAVE-V1",
    "SIGLIP-LSTM-V3": "SIGLIP-LSTM-V3", // Legacy
    "SIGLIP-LSTM-V1": "SIGLIP-LSTM-V1", // Legacy
};

/** Metadata for displaying models in the UI */
export const MODEL_INFO = {
    [MODEL_TYPES["SIGLIP-LSTM-V4"]]: {
        label: "SigLip LSTM v4",
        description: "State-of-the-art temporal analysis model with dropout.",
    },
    [MODEL_TYPES["EFFICIENTNET-B7-V1"]]: {
        label: "EfficientNet B7",
        description: "High-accuracy frame-by-frame face classifier.",
    },
    [MODEL_TYPES["COLOR-CUES-LSTM-V1"]]: {
        label: "Color Cues LSTM v1",
        description: "Specialized model focusing on color inconsistencies.",
    },
    [MODEL_TYPES["EYEBLINK-CNN-LSTM-V1"]]: {
        label: "Eyeblink CNN+LSTM",
        description: "Detects inconsistencies in eye blinking patterns.",
    },
    [MODEL_TYPES["SCATTERING-WAVE-V1"]]: {
        label: "Scattering Wave v1",
        description: "Analyzes audio tracks for signs of voice cloning.",
    },
    [MODEL_TYPES["SIGLIP-LSTM-V3"]]: {
        label: "SigLip LSTM v3 (Legacy)",
        description: "Previous-generation temporal analysis model.",
    },
    [MODEL_TYPES["SIGLIP-LSTM-V1"]]: {
        label: "SigLip LSTM v1 (Legacy)",
        description: "The foundational temporal analysis model.",
    },
};

// ADD THIS MISSING EXPORT BACK
/** Metadata for displaying analysis types in the UI */
export const ANALYSIS_TYPE_INFO = {
    [ANALYSIS_TYPES.COMPREHENSIVE]: {
        label: "Comprehensive Analysis",
        description:
            "Complete analysis with detailed metrics and confidence scores.",
    },
    // Add other types here if they become individually selectable in the future
};