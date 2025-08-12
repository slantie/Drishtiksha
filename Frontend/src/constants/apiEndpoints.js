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
};
