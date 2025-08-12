// src/constants/apiEndpoints.js
/**
 * @fileoverview Centralized API endpoints for the application.
 */

const API_BASE_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:4000";

export const API_ENDPOINTS = {
    // Video Routes
    GET_VIDEOS: `${API_BASE_URL}/api/video`,
    UPLOAD_VIDEO: `${API_BASE_URL}/api/video/upload`,
    
    /**
     * @param {string} videoId
     * @returns {string}
     */
    ANALYZE_VIDEO: (videoId) => `${API_BASE_URL}/api/video/${videoId}/analyze`,

    /**
     * @param {string} videoId
     * @returns {string}
     */
    UPDATE_VIDEO: (videoId) => `${API_BASE_URL}/api/video/${videoId}`,

    /**
     * @param {string} videoId
     * @returns {string}
     */
    DELETE_VIDEO: (videoId) => `${API_BASE_URL}/api/video/${videoId}`,
};
