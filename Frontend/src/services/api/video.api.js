// src/services/api/video.api.js

import axiosInstance from "../../lib/axios.js";
import { API_ENDPOINTS } from "../../constants/apiEndpoints.js";

export const videoApi = {
    /**
     * Fetches all videos for the authenticated user.
     * @returns {Promise<object>} The API response containing an array of videos.
     */
    getAll: async () => {
        return await axiosInstance.get(API_ENDPOINTS.VIDEOS.ALL);
    },

    /**
     * Fetches a single video by its unique ID, including all its analysis results.
     * @param {string} videoId - The ID of the video.
     * @returns {Promise<object>} The API response containing the detailed video object.
     */
    getById: async (videoId) => {
        return await axiosInstance.get(API_ENDPOINTS.VIDEOS.BY_ID(videoId));
    },

    /**
     * Uploads a new video file and its metadata.
     * @param {FormData} formData - The FormData object containing the video file and description.
     * @returns {Promise<object>} The API response with the newly created video record.
     */
    upload: async (formData) => {
        return await axiosInstance.post(API_ENDPOINTS.VIDEOS.ALL, formData, {
            timeout: 300000, // 5 minutes
        });
    },

    /**
     * Updates a video's metadata (e.g., filename, description).
     * @param {string} videoId - The ID of the video to update.
     * @param {object} updateData - An object containing the fields to update.
     * @returns {Promise<object>} The API response with the updated video data.
     */
    update: async (videoId, updateData) => {
        console.log("Updating video:", videoId, "with data:", updateData); // Debug log
        return await axiosInstance.patch(
            API_ENDPOINTS.VIDEOS.BY_ID(videoId),
            updateData
        );
    },

    /**
     * Deletes a video and all its associated data.
     * @param {string} videoId - The ID of the video to delete.
     * @returns {Promise<object>} The API response confirming the deletion.
     */
    delete: async (videoId) => {
        return await axiosInstance.delete(API_ENDPOINTS.VIDEOS.BY_ID(videoId));
    },

    /**
     * Triggers a new, manual analysis for a specific video.
     * NOTE: This requires a corresponding backend endpoint (e.g., POST /videos/:id/analyze)
     * which is not yet present in the backend code. This is a placeholder.
     * @param {string} videoId - The ID of the video.
     * @param {object} analysisConfig - { type, model }
     * @returns {Promise<object>} The API response.
     */
    createAnalysis: async (videoId, analysisConfig) => {
        console.warn(
            "Frontend is calling createAnalysis, but the backend endpoint is not yet implemented. Simulating success."
        );
        // In a real scenario, the following line would be used:
        // return await axiosInstance.post(`/videos/${videoId}/analyze`, analysisConfig);
        return Promise.resolve({
            success: true,
            message: "Manual analysis queued successfully (simulated).",
        });
    },
};
