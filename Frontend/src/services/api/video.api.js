// src/services/api/video.api.js

import axiosInstance from "../../lib/axios.js";

export const videoApi = {
    /**
     * Get all videos for the authenticated user
     * @returns {Promise} Response with videos array
     */
    getAllVideos: async () => {
        return await axiosInstance.get("/videos");
    },

    /**
     * Get a specific video by ID
     * @param {string} videoId - The video ID
     * @returns {Promise} Response with video data
     */
    getVideoById: async (videoId) => {
        return await axiosInstance.get(`/videos/${videoId}`);
    },

    /**
     * Upload a new video
     * @param {FormData} formData - Form data containing video file and metadata
     * @returns {Promise} Response with uploaded video data
     */
    uploadVideo: async (formData) => {
        return await axiosInstance.post("/videos", formData);
    },

    /**
     * Update video metadata
     * @param {string} videoId - The video ID
     * @param {Object} updateData - Data to update
     * @returns {Promise} Response with updated video data
     */
    updateVideo: async (videoId, updateData) => {
        return await axiosInstance.patch(`/videos/${videoId}`, updateData);
    },

    /**
     * Triggers the generation of a visualized analysis video.
     * @param {string} videoId - The ID of the video to analyze.
     * @returns {Promise} Response with the updated video data, including the new visualizedUrl.
     */
    generateVisualAnalysis: async (videoId) => {
        // This is a long-running operation, so we increase the timeout specifically for it.
        return await axiosInstance.post(`/videos/${videoId}/visualize`, {}, {
            timeout: 600000, // 10 minute timeout
        });
    },

    /**
     * Delete a video
     * @param {string} videoId - The video ID
     * @returns {Promise} Response confirming deletion
     */
    deleteVideo: async (videoId) => {
        return await axiosInstance.delete(`/videos/${videoId}`);
    },
};
