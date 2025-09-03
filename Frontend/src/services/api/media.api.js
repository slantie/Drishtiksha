// src/services/api/media.api.js

import axiosInstance from "../../lib/axios.js";

const MEDIA_ROUTES = {
  BASE: "/api/v1/media", // Explicitly include API versioning
  BY_ID: (id) => `/api/v1/media/${id}`, // Explicitly include API versioning
  RERUN_ANALYSIS: (id) => `/api/v1/media/${id}/analyze`, // Explicitly include API versioning
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
   * The `formData` should contain the file under the key 'media'.
   */
  upload: async (formData) => {
    // The axiosInstance already handles Content-Type for FormData,
    // but explicitly setting it to undefined here for clarity that it's multipart
    // and using a longer timeout for file uploads.
    return axiosInstance.post(MEDIA_ROUTES.BASE, formData, {
      timeout: 300000, // 5 minutes for large files
      headers: {
        "Content-Type": "multipart/form-data", // Explicitly set for clarity
      },
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
   * Triggers a new analysis run for an existing media item.
   */
  rerunAnalysis: async (mediaId) => {
    return axiosInstance.post(MEDIA_ROUTES.RERUN_ANALYSIS(mediaId));
  },
};
