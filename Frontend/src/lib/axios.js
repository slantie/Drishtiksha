// src/lib/axios.js

import axios from "axios";
import { config } from "../config/env.js"; // Ensure this import is explicit
import { authStorage } from "../utils/authStorage.js"; // Ensure this import is explicit
import queryClient from "./queryClient.js"; // Ensure this import is explicit

const axiosInstance = axios.create({
  baseURL: config.VITE_BACKEND_URL, // Use the validated config directly
  timeout: 60000,
  withCredentials: true, // Key addition: Enables sending/receiving cookies cross-origin
});

axiosInstance.interceptors.request.use(
  (reqConfig) => {
    const token = authStorage.get().token;
    if (token) {
      reqConfig.headers.Authorization = `Bearer ${token}`;
    }

    // Handle Content-Type for FormData correctly: Axios automatically sets
    // multipart/form-data with the correct boundary when FormData is used.
    // Explicitly setting application/json for other requests is a good practice.
    if (!(reqConfig.data instanceof FormData)) {
      reqConfig.headers["Content-Type"] = "application/json";
    }
    // If it's FormData, let Axios handle the Content-Type automatically.
    // We don't need to delete it manually.

    return reqConfig;
  },
  (error) => Promise.reject(error)
);

axiosInstance.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const defaultError = "An unexpected error occurred.";
    if (error.response) {
      // More robust 401 handling: Clear cache and redirect
      if (
        error.response.status === 401 &&
        !window.location.pathname.includes("/auth")
      ) {
        authStorage.clear();
        queryClient.clear(); // Clear all React Query cache on 401
        // Redirect to auth page, preserving the original path for redirection after login
        window.location.href = `/auth?session_expired=true&redirect=${encodeURIComponent(
          window.location.pathname
        )}`;
        // Prevent further processing of this error
        return new Promise(() => {}); // Return a never-resolving promise
      }

      // Extract more specific error messages from backend responses
      const errorMessage =
        error.response.data?.message ||
        error.response.data?.error?.message || // Check for nested error objects
        error.response.data?.error ||
        defaultError;
      return Promise.reject(new Error(errorMessage));
    } else if (error.request) {
      // The request was made but no response was received (e.g., network down)
      return Promise.reject(
        new Error("Network error. The server could not be reached.")
      );
    } else {
      // Something else happened while setting up the request
      return Promise.reject(new Error(error.message || defaultError));
    }
  }
);

export default axiosInstance;
