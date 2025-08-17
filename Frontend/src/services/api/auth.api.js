// src/services/api/auth.api.js

import axiosInstance from "../../lib/axios.js";
import { API_ENDPOINTS } from "../../constants/apiEndpoints.js";

export const authApi = {
    /**
     * Logs in a user with their credentials.
     * @param {object} credentials - { email, password }
     * @returns {Promise<object>} The API response containing user data and token.
     */
    login: async (credentials) => {
        return await axiosInstance.post(API_ENDPOINTS.AUTH.LOGIN, credentials);
    },

    /**
     * Registers a new user.
     * @param {object} userData - { firstName, lastName, email, password }
     * @returns {Promise<object>} The API response with the newly created user data.
     */
    signup: async (userData) => {
        return await axiosInstance.post(API_ENDPOINTS.AUTH.SIGNUP, userData);
    },

    /**
     * Retrieves the profile of the currently authenticated user.
     * @returns {Promise<object>} The API response with the user's profile information.
     */
    getProfile: async () => {
        return await axiosInstance.get(API_ENDPOINTS.AUTH.PROFILE);
    },

    /**
     * Updates the profile of the currently authenticated user.
     * @param {object} profileData - The fields to update (e.g., { firstName, bio }).
     * @returns {Promise<object>} The API response with the updated user profile.
     */
    updateProfile: async (profileData) => {
        return await axiosInstance.put(API_ENDPOINTS.AUTH.PROFILE, profileData);
    },

    /**
     * Updates the password for the currently authenticated user.
     * @param {object} passwordData - { currentPassword, newPassword }
     * @returns {Promise<object>} The API response confirming the update.
     */
    updatePassword: async (passwordData) => {
        return await axiosInstance.put(
            API_ENDPOINTS.AUTH.UPDATE_PASSWORD,
            passwordData
        );
    },

    /**
     * Updates the avatar URL for the currently authenticated user.
     * @param {object} avatarData - { avatar: "http://new-avatar-url.com/img.png" }
     * @returns {Promise<object>} The API response with the updated user profile.
     */
    updateAvatar: async (avatarData) => {
        return await axiosInstance.put(
            API_ENDPOINTS.AUTH.UPDATE_AVATAR,
            avatarData
        );
    },

    /**
     * Deletes the avatar for the currently authenticated user.
     * @returns {Promise<object>} The API response with the updated user profile.
     */
    deleteAvatar: async () => {
        return await axiosInstance.delete(API_ENDPOINTS.AUTH.UPDATE_AVATAR);
    },

    /**
     * Logs the user out by clearing client-side storage.
     * The backend is stateless, so no API call is needed.
     * @returns {Promise<void>}
     */
    logout: async () => {
        sessionStorage.removeItem("authToken");
        sessionStorage.removeItem("user");
        return Promise.resolve();
    },
};
