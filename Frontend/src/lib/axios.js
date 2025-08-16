// src/lib/axios.js

import axios from "axios";
import { API_BASE_URL } from "../constants/apiEndpoints.js";

const API_VERSION = "/api/v1";

const axiosInstance = axios.create({
    baseURL: `${API_BASE_URL}${API_VERSION}`,
    timeout: 60000, // 60 seconds timeout
    headers: {
        "Content-Type": "application/json",
    },
});

// Request interceptor to dynamically add the auth token to every request
axiosInstance.interceptors.request.use(
    (config) => {
        const token =
            localStorage.getItem("authToken") ||
            sessionStorage.getItem("authToken");

        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }

        // For file uploads, let the browser set the Content-Type with the correct boundary
        if (config.data instanceof FormData) {
            delete config.headers["Content-Type"];
        }

        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

// Response interceptor to handle global errors and response structures
axiosInstance.interceptors.response.use(
    (response) => {
        // The backend wraps successful responses in a standard format.
        // We return the inner `data` object for convenience in React Query.
        return response.data;
    },
    (error) => {
        const defaultError = "An unexpected error occurred.";

        if (error.response) {
            // The request was made and the server responded with a status code
            // that falls out of the range of 2xx
            const { status, data } = error.response;

            // Handle 401 Unauthorized: Clear session and redirect to login
            if (status === 401) {
                localStorage.removeItem("authToken");
                localStorage.removeItem("user");
                sessionStorage.removeItem("authToken");
                sessionStorage.removeItem("user");

                if (!window.location.pathname.includes("/auth")) {
                    window.location.href = "/auth?session_expired=true";
                }
            }

            // Use the specific error message from the backend's ApiError response
            const errorMessage = data?.message || data?.error || defaultError;
            return Promise.reject(new Error(errorMessage));
        } else if (error.request) {
            // The request was made but no response was received
            return Promise.reject(
                new Error("Network error. Please check your connection.")
            );
        } else {
            // Something happened in setting up the request that triggered an Error
            return Promise.reject(new Error(error.message || defaultError));
        }
    }
);

export default axiosInstance;
