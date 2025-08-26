// src/services/api/media.api.js

import axiosInstance from "../../lib/axios.js";
import { API_ENDPOINTS } from "../../constants/apiEndpoints.js";

// REFACTORED: Renamed from 'videoApi' to 'mediaApi'
export const mediaApi = {
    /**
     * Fetches all media for the authenticated user.
     * @returns {Promise<object>} The API response containing an array of media items.
     */
    getAll: async () => {
        // UPDATED: Using the new MEDIA constant
        return await axiosInstance.get(API_ENDPOINTS.MEDIA.ALL);
    },

    /**
     * Fetches a single media item by its unique ID, including all its analysis results.
     * @param {string} mediaId - The ID of the media item.
     * @returns {Promise<object>} The API response containing the detailed media object.
     */
    getById: async (mediaId) => {
        // UPDATED: Using the new MEDIA constant
        return await axiosInstance.get(API_ENDPOINTS.MEDIA.BY_ID(mediaId));
    },

    /**
     * Uploads a new media file (video, audio, etc.) and its metadata.
     * @param {FormData} formData - The FormData object containing the file and description.
     *                            NOTE: The file field MUST be named 'file'.
     * @returns {Promise<object>} The API response with the newly created media record.
     */
    upload: async (formData) => {
        // UPDATED: Using the new MEDIA constant
        return await axiosInstance.post(API_ENDPOINTS.MEDIA.ALL, formData, {
            timeout: 300000, // 5 minutes for potentially large uploads
        });
    },

    /**
     * Updates a media item's metadata (e.g., filename, description).
     * @param {string} mediaId - The ID of the media to update.
     * @param {object} updateData - An object containing the fields to update.
     * @returns {Promise<object>} The API response with the updated media data.
     */
    update: async (mediaId, updateData) => {
        // UPDATED: Using the new MEDIA constant
        return await axiosInstance.patch(
            API_ENDPOINTS.MEDIA.BY_ID(mediaId),
            updateData
        );
    },

    /**
     * Deletes a media item and all its associated data.
     * @param {string} mediaId - The ID of the media to delete.
     * @returns {Promise<object>} The API response confirming the deletion.
     */
    delete: async (mediaId) => {
        // UPDATED: Using the new MEDIA constant
        return await axiosInstance.delete(API_ENDPOINTS.MEDIA.BY_ID(mediaId));
    },

    /**
     * Triggers a new, manual analysis for a specific media item.
     * @param {string} mediaId - The ID of the media item.
     * @param {object} analysisConfig - { model: "MODEL_NAME" }
     * @returns {Promise<object>} The API response.
     */
    createAnalysis: async (mediaId, analysisConfig) => {
        // NOTE: This endpoint might need to be updated to /media/:id/analyze if the backend changes.
        // For now, the placeholder logic is preserved.
        console.warn(
            "Frontend is calling createAnalysis, which may be a legacy endpoint. Simulating success."
        );
        return Promise.resolve({
            success: true,
            message: "Manual analysis queued successfully (simulated).",
        });
    },
};