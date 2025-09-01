// src/lib/axios.js

import axios from "axios";
import { config } from "../config/env.js";
import { authStorage } from "../utils/authStorage.js";

const axiosInstance = axios.create({
  baseURL: `${config.VITE_BACKEND_URL}${config.VITE_BACKEND_URL_VERSION}`,
  timeout: 60000,
});

axiosInstance.interceptors.request.use(
  (config) => {
    const token = authStorage.get().token;
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    if (config.data instanceof FormData) {
      delete config.headers["Content-Type"];
    } else {
      config.headers["Content-Type"] = "application/json";
    }
    return config;
  },
  (error) => Promise.reject(error)
);

axiosInstance.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const defaultError = "An unexpected error occurred.";
    if (error.response) {
      // FIX: More robust 401 handling to prevent redirect loops.
      if (
        error.response.status === 401 &&
        !window.location.pathname.includes("/auth")
      ) {
        authStorage.clear();
        window.location.href = `/auth?session_expired=true&redirect=${encodeURIComponent(
          window.location.pathname
        )}`;
      }
      const errorMessage =
        error.response.data?.message ||
        error.response.data?.error ||
        defaultError;
      return Promise.reject(new Error(errorMessage));
    } else if (error.request) {
      return Promise.reject(
        new Error("Network error. The server could not be reached.")
      );
    }
    return Promise.reject(new Error(error.message || defaultError));
  }
);

export default axiosInstance;
