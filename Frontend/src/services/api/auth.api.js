// src/services/api/auth.api.js

import axiosInstance from "../../lib/axios.js";

export const authApi = {
    /**
     * Login user
     * @param {Object} credentials - Login credentials
     * @param {string} credentials.email - User email
     * @param {string} credentials.password - User password
     * @returns {Promise} Response with user data and token
     */
    login: async (credentials) => {
        return await axiosInstance.post("/auth/login", credentials);
    },

    /**
     * Register new user
     * @param {Object} userData - User registration data
     * @returns {Promise} Response with created user data
     */
    signup: async (userData) => {
        return await axiosInstance.post("/auth/signup", userData);
    },

    /**
     * Get user profile
     * @returns {Promise} Response with user profile data
     */
    getProfile: async () => {
        return await axiosInstance.get("/auth/profile");
    },

    /**
     * Update user profile
     * @param {Object} profileData - Updated profile data
     * @returns {Promise} Response with updated user data
     */
    updateProfile: async (profileData) => {
        return await axiosInstance.put("/auth/profile", profileData);
    },

    /**
     * Update user password
     * @param {Object} passwordData - Password update data
     * @param {string} passwordData.currentPassword - Current password
     * @param {string} passwordData.newPassword - New password
     * @returns {Promise} Response confirming password update
     */
    updatePassword: async (passwordData) => {
        return await axiosInstance.put("/auth/profile/password", passwordData);
    },

    /**
     * Update user avatar
     * @param {Object} avatarData - Avatar data with URL
     * @param {string} avatarData.avatar - Avatar URL
     * @returns {Promise} Response with updated avatar URL
     */
    updateAvatar: async (avatarData) => {
        return await axiosInstance.put("/auth/profile/avatar", avatarData);
    },

    /**
     * Delete user avatar
     * @returns {Promise} Response confirming avatar deletion
     */
    deleteAvatar: async () => {
        return await axiosInstance.delete("/auth/profile/avatar");
    },

    /**
     * Logout user (client-side only)
     * @returns {Promise} Resolved promise
     */
    logout: async () => {
        // Clear local storage
        localStorage.removeItem("authToken");
        localStorage.removeItem("user");
        sessionStorage.removeItem("authToken");
        sessionStorage.removeItem("user");

        return Promise.resolve();
    },
};
