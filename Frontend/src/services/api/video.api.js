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
     * @param {string} model - Optional specific model to use for visualization
     * @returns {Promise} Response with the updated video data, including the new visualizedUrl.
     */
    generateVisualAnalysis: async (videoId, model = null) => {
        const payload = model ? { model } : {};
        return await axiosInstance.post(
            `/videos/${videoId}/visualize`,
            payload,
            {
                timeout: 600000, // 10 minute timeout
            }
        );
    },

    /**
     * Delete a video
     * @param {string} videoId - The video ID
     * @returns {Promise} Response confirming deletion
     */
    deleteVideo: async (videoId) => {
        return await axiosInstance.delete(`/videos/${videoId}`);
    },

    // --- Enhanced Analysis API Methods ---

    /**
     * Get model service status and available models
     * @returns {Promise} Response with model status information
     */
    getModelStatus: async () => {
        return await axiosInstance.get("/videos/model/status");
    },

    /**
     * Create a specific analysis for a video
     * @param {string} videoId - The video ID
     * @param {string} type - Analysis type (QUICK, DETAILED, FRAMES, VISUALIZE)
     * @param {string} model - Model to use (SIGLIP_LSTM_V1, SIGLIP_LSTM_V3, COLOR_CUES_LSTM_V1)
     * @returns {Promise} Response with created analysis data
     */
    createAnalysis: async (videoId, type, model) => {
        return await axiosInstance.post(
            `/videos/${videoId}/analyze`,
            {
                type,
                model,
            },
            {
                timeout: 300000, // 5 minute timeout for analysis
            }
        );
    },

    /**
     * Get analysis results for a video with optional filtering
     * @param {string} videoId - The video ID
     * @param {Object} filters - Optional filters
     * @param {string} filters.type - Filter by analysis type
     * @param {string} filters.model - Filter by model
     * @returns {Promise} Response with analysis results
     */
    getAnalysisResults: async (videoId, filters = {}) => {
        const params = new URLSearchParams();
        if (filters.type) params.append("type", filters.type);
        if (filters.model) params.append("model", filters.model);

        const queryString = params.toString();
        const url = queryString
            ? `/videos/${videoId}/analysis?${queryString}`
            : `/videos/${videoId}/analysis`;

        return await axiosInstance.get(url);
    },

    /**
     * Get specific analysis by type and model
     * @param {string} videoId - The video ID
     * @param {string} type - Analysis type
     * @param {string} model - Model name
     * @returns {Promise} Response with specific analysis data
     */
    getSpecificAnalysis: async (videoId, type, model) => {
        return await axiosInstance.get(
            `/videos/${videoId}/analysis?type=${type}&model=${model}`
        );
    },

    /**
     * Create multiple analyses for a video (batch creation)
     * @param {string} videoId - The video ID
     * @param {Array} analysisConfigs - Array of {type, model} configurations
     * @returns {Promise} Response with created analyses
     */
    createMultipleAnalyses: async (videoId, analysisConfigs) => {
        const promises = analysisConfigs.map((config) =>
            videoApi.createAnalysis(videoId, config.type, config.model)
        );
        return Promise.all(promises);
    },

    /**
     * Create visualization with specific model
     * @param {string} videoId - The video ID
     * @param {string} model - Specific model for visualization
     * @returns {Promise} Response with visualization data
     */
    createModelVisualization: async (videoId, model) => {
        return await axiosInstance.post(
            `/videos/${videoId}/visualize`,
            { model },
            {
                timeout: 600000, // 10 minute timeout
            }
        );
    },
};
