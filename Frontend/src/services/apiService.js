// src/services/apiService.js

import { API_ENDPOINTS } from "../constants/apiEndpoints.js";

/**
 * A helper function to handle API requests and responses.
 * @param {string} url - The URL to fetch.
 * @param {object} options - The options for the fetch request (method, body, headers).
 * @returns {Promise<object>} - The JSON response from the server.
 */
const apiRequest = async (url, options = {}) => {
    const token =
        localStorage.getItem("authToken") ||
        sessionStorage.getItem("authToken");

    const headers = {
        "Content-Type": "application/json",
        ...options.headers,
    };

    if (token) {
        headers["Authorization"] = `Bearer ${token}`;
    }

    try {
        const response = await fetch(url, { ...options, headers });
        const data = await response.json();

        if (!response.ok) {
            // Use the server's error message if available, otherwise use a default
            const errorMessage =
                data.message || `Request failed with status ${response.status}`;
            throw new Error(errorMessage);
        }

        return data;
    } catch (error) {
        console.error(`API request to ${url} failed:`, error);
        // Re-throw the error so it can be caught by the calling function
        throw error;
    }
};

export const authApiService = {
    /**
     * Logs in a user.
     * @param {string} email - The user's email.
     * @param {string} password - The user's password.
     * @returns {Promise<object>} - The user and token data.
     */
    login: (email, password) => {
        return apiRequest(API_ENDPOINTS.LOGIN, {
            method: "POST",
            body: JSON.stringify({ email, password }),
        });
    },

    /**
     * Signs up a new user.
     * @param {object} signupData - The user's registration data.
     * @returns {Promise<object>} - The newly created user data.
     */
    signup: (signupData) => {
        return apiRequest(API_ENDPOINTS.SIGNUP, {
            method: "POST",
            body: JSON.stringify(signupData),
        });
    },
};
