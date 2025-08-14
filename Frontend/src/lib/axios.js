// src/lib/axios.js

import axios from "axios";
import { API_ENDPOINTS } from "../constants/apiEndpoints.js";

const API_BASE_URL =
    import.meta.env.VITE_BACKEND_URL || "http://localhost:3000";
const API_VERSION = "/api/v1";

// Create axios instance
const axiosInstance = axios.create({
    baseURL: `${API_BASE_URL}${API_VERSION}`,
    timeout: 30000, // 30 seconds timeout
    headers: {
        "Content-Type": "application/json",
    },
});

// Request interceptor to add authentication token
axiosInstance.interceptors.request.use(
    (config) => {
        const token =
            localStorage.getItem("authToken") ||
            sessionStorage.getItem("authToken");

        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }

        // Handle FormData - don't set Content-Type for file uploads
        if (config.data instanceof FormData) {
            delete config.headers["Content-Type"];
        }

        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

// Response interceptor to handle common responses and errors
axiosInstance.interceptors.response.use(
    (response) => {
        // Return the data directly for successful responses
        return response.data;
    },
    (error) => {
        // Handle different error types
        if (error.response) {
            // Server responded with error status
            const { status, data } = error.response;

            // Handle unauthorized access
            if (status === 401) {
                // Clear tokens and redirect to login
                localStorage.removeItem("authToken");
                localStorage.removeItem("user");
                sessionStorage.removeItem("authToken");
                sessionStorage.removeItem("user");

                // Only redirect if not already on auth page
                if (!window.location.pathname.includes("/auth")) {
                    window.location.href = "/auth";
                }
            }

            // Throw error with server message or default message
            const errorMessage =
                data?.message ||
                data?.error ||
                `Request failed with status ${status}`;
            throw new Error(errorMessage);
        } else if (error.request) {
            // Network error
            throw new Error(
                "Network error. Please check your connection and try again."
            );
        } else {
            // Something else happened
            throw new Error("An unexpected error occurred.");
        }
    }
);

export default axiosInstance;
