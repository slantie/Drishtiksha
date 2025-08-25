// src/services/videoApiService.js

import { API_ENDPOINTS } from "../constants/apiEndpoints.js";

const getAuthToken = () =>
    localStorage.getItem("authToken") || sessionStorage.getItem("authToken");

const apiRequest = async (url, options = {}) => {
    const token = getAuthToken();
    const headers = { ...options.headers };
    if (token) {
        headers["Authorization"] = `Bearer ${token}`;
    }
    if (!(options.body instanceof FormData)) {
        headers["Content-Type"] = "application/json";
    }
    try {
        const response = await fetch(url, { ...options, headers });
        // Handle cases where the response might not be JSON
        const contentType = response.headers.get("content-type");
        if (contentType && contentType.indexOf("application/json") !== -1) {
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.message || "API request failed");
            }
            return data;
        } else {
            const text = await response.text();
            throw new Error(
                `Expected JSON but received: ${text.substring(0, 100)}...`
            );
        }
    } catch (error) {
        console.error(`API Error fetching ${url}:`, error);
        throw error;
    }
};

export const videoApiService = {
    getAllVideos: () => apiRequest(API_ENDPOINTS.VIDEOS),
    getVideoById: (videoId) => apiRequest(API_ENDPOINTS.VIDEO_BY_ID(videoId)),
    uploadVideo: (formData) =>
        apiRequest(API_ENDPOINTS.VIDEOS, { method: "POST", body: formData }),
    updateVideo: (videoId, updateData) =>
        apiRequest(API_ENDPOINTS.VIDEO_BY_ID(videoId), {
            method: "PATCH",
            body: JSON.stringify(updateData),
        }),
    deleteVideo: (videoId) =>
        apiRequest(API_ENDPOINTS.VIDEO_BY_ID(videoId), { method: "DELETE" }),
};
