// src/constants/apiEndpoints.js
/**
 * @fileoverview Centralized API endpoints for the application.
 * This file contains all the API endpoints used to communicate with the backend.
 */

// The base URL for the backend API, configured via environment variables.
const API_BASE_URL =
    import.meta.env.VITE_BACKEND_URL || "http://localhost:3000";
const API_VERSION = "/api/v1";

export const API_ENDPOINTS = {
    // --- Authentication Routes ---
    AUTH_BASE: `${API_BASE_URL}${API_VERSION}/auth`,

    /** Registers a new user. */
    SIGNUP: `${API_BASE_URL}${API_VERSION}/auth/signup`,

    /** Logs in an existing user. */
    LOGIN: `${API_BASE_URL}${API_VERSION}/auth/login`,

    /** Logs out a user (client-side token deletion). */
    LOGOUT: `${API_BASE_URL}${API_VERSION}/auth/logout`,

    /** Retrieves the authenticated user's profile. */
    GET_PROFILE: `${API_BASE_URL}${API_VERSION}/auth/profile`,

    /** Updates the authenticated user's profile information. */
    UPDATE_PROFILE: `${API_BASE_URL}${API_VERSION}/auth/profile`,

    /** Updates the authenticated user's password. */
    UPDATE_PASSWORD: `${API_BASE_URL}${API_VERSION}/auth/profile/password`,

    /** Updates the authenticated user's avatar. */
    UPDATE_AVATAR: `${API_BASE_URL}${API_VERSION}/auth/profile/avatar`,

    /** Deletes the authenticated user's avatar. */
    DELETE_AVATAR: `${API_BASE_URL}${API_VERSION}/auth/profile/avatar`,

    // --- Video Routes ---
    VIDEOS_BASE: `${API_BASE_URL}${API_VERSION}/videos`,

    /** * Retrieves all videos for the user.
     * Also used for uploading a new video (POST).
     */
    VIDEOS: `${API_BASE_URL}${API_VERSION}/videos`,

    /**
     * Generates a URL for a specific video resource.
     * Used for GET (retrieve), PATCH (update), and DELETE operations.
     * @param {string} videoId - The unique identifier of the video.
     * @returns {string} The full API URL for the specified video.
     */
    VIDEO_BY_ID: (videoId) => `${API_BASE_URL}${API_VERSION}/videos/${videoId}`,

    // --- Enhanced Analysis Routes ---

    /** Gets model service status and available models */
    MODEL_STATUS: `${API_BASE_URL}${API_VERSION}/videos/status`,

    /** Gets available models from the ML server */
    AVAILABLE_MODELS: `${API_BASE_URL}${API_VERSION}/videos/models`,

    /**
     * Creates a specific analysis for a video
     * @param {string} videoId - The video ID
     * @returns {string} The full API URL for creating analysis
     */
    CREATE_ANALYSIS: (videoId) =>
        `${API_BASE_URL}${API_VERSION}/videos/${videoId}/analyze`,

    /**
     * Creates visualization analysis for a video
     * @param {string} videoId - The video ID
     * @returns {string} The full API URL for creating visualization
     */
    CREATE_VISUALIZATION: (videoId) =>
        `${API_BASE_URL}${API_VERSION}/videos/${videoId}/visualize`,

    /**
     * Gets analysis results for a video (included in video details)
     * @param {string} videoId - The video ID
     * @returns {string} The full API URL for video details with analysis results
     */
    GET_ANALYSIS_RESULTS: (videoId) =>
        `${API_BASE_URL}${API_VERSION}/videos/${videoId}`,

    /**
     * Gets specific analysis by type and model (client-side filtering)
     * Note: This now uses the video details endpoint with client-side filtering
     * @param {string} videoId - The video ID
     * @param {string} _TYPE - Analysis type (QUICK, DETAILED, FRAMES, VISUALIZE) - unused, kept for compatibility
     * @param {string} _MODEL - Model name (SIGLIP_LSTM_V1, SIGLIP_LSTM_V3, COLOR_CUES_LSTM_V1) - unused, kept for compatibility
     * @returns {string} The full API URL for video details
     */
    GET_SPECIFIC_ANALYSIS: (videoId, _TYPE, _MODEL) =>
        `${API_BASE_URL}${API_VERSION}/videos/${videoId}`,
};

// --- Constants for Enhanced Analysis System ---

/** Supported analysis types */
export const ANALYSIS_TYPES = {
    QUICK: "QUICK",
    FRAMES: "FRAMES",
    VISUALIZE: "VISUALIZE",
    COMPREHENSIVE: "COMPREHENSIVE", // New merged endpoint
};

/** Supported model types */
export const MODEL_TYPES = {
    SIGLIP_LSTM_V1: "SIGLIP_LSTM_V1",
    SIGLIP_LSTM_V3: "SIGLIP_LSTM_V3",
    COLOR_CUES_LSTM_V1: "COLOR_CUES_LSTM_V1",
};

/** Analysis type metadata */
export const ANALYSIS_TYPE_INFO = {
    [ANALYSIS_TYPES.QUICK]: {
        label: "Comprehensive Analysis",
        description:
            "Complete deepfake analysis with detailed metrics and confidence scores",
        duration: "~30 seconds - 2 minutes",
        icon: "🔍",
    },
    [ANALYSIS_TYPES.FRAMES]: {
        label: "Frame-by-Frame Analysis",
        description:
            "Per-frame analysis with temporal data and detailed breakdowns",
        duration: "~3-5 minutes",
        icon: "🎞️",
    },
    [ANALYSIS_TYPES.VISUALIZE]: {
        label: "Visual Analysis",
        description:
            "Generate annotated video with analysis visualization overlay",
        duration: "~5-10 minutes",
        icon: "📊",
    },
    [ANALYSIS_TYPES.COMPREHENSIVE]: {
        label: "Complete Analysis Suite",
        description:
            "All analysis types in one request: comprehensive + frames + visualization",
        duration: "~5-10 minutes",
        icon: "🚀",
    },
};

/** Model metadata */
export const MODEL_INFO = {
    [MODEL_TYPES.SIGLIP_LSTM_V1]: {
        label: "SigLIP LSTM v1",
        description: "Primary deepfake detection model with high accuracy",
        version: "1.0.0",
        specialty: "General deepfake detection",
    },
    [MODEL_TYPES.SIGLIP_LSTM_V3]: {
        label: "SigLIP LSTM v3",
        description: "Enhanced version with improved accuracy and speed",
        version: "3.0.0",
        specialty: "High-accuracy detection",
    },
    [MODEL_TYPES.COLOR_CUES_LSTM_V1]: {
        label: "Color Cues LSTM v1",
        description: "Specialized model focusing on color inconsistencies",
        version: "1.0.0",
        specialty: "Color-based detection",
    },
};
