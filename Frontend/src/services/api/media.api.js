// src/services/api/media.api.js

import axiosInstance from '../../lib/axios.js';

const MEDIA_ROUTES = {
    BASE: '/media',
    BY_ID: (id) => `/media/${id}`,
    RERUN_ANALYSIS: (id) => `/media/${id}/analyze`,
};

export const mediaApi = {
    /**
     * Fetches all media items for the authenticated user.
     */
    getAll: async () => axiosInstance.get(MEDIA_ROUTES.BASE),

    /**
     * Fetches a single media item by its ID, including all historical analysis runs.
     */
    getById: async (mediaId) => axiosInstance.get(MEDIA_ROUTES.BY_ID(mediaId)),

    /**
     * Uploads a new media file to initiate the first analysis run.
     */
    upload: async (formData) => {
        return axiosInstance.post(MEDIA_ROUTES.BASE, formData, {
            // Use a longer timeout for file uploads.
            timeout: 300000, // 5 minutes
        });
    },

    /**
     * Updates a media item's metadata (e.g., description).
     */
    update: async (mediaId, updateData) => {
        return axiosInstance.patch(MEDIA_ROUTES.BY_ID(mediaId), updateData);
    },

    /**
     * Deletes a media item and all its associated data.
     */
    delete: async (mediaId) => {
        return axiosInstance.delete(MEDIA_ROUTES.BY_ID(mediaId));
    },

    /**
     * NEW: Triggers a new analysis run for an existing media item.
     */
    rerunAnalysis: async (mediaId) => {
        return axiosInstance.post(MEDIA_ROUTES.RERUN_ANALYSIS(mediaId));
    },
};