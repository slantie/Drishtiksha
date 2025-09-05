// src/services/api/auth.api.js

import axiosInstance from "../../lib/axios.js";

const AUTH_ROUTES = {
  LOGIN: "/api/v1/auth/login", // Explicitly include API versioning
  SIGNUP: "/api/v1/auth/signup", // Explicitly include API versioning
  LOGOUT: "/api/v1/auth/logout", // Explicitly include API versioning
  PROFILE: "/api/v1/auth/profile", // Explicitly include API versioning
  UPDATE_PASSWORD: "/api/v1/auth/profile/password", // Explicitly include API versioning
};

export const authApi = {
  login: async (credentials) =>
    axiosInstance.post(AUTH_ROUTES.LOGIN, credentials),
  signup: async (userData) => axiosInstance.post(AUTH_ROUTES.SIGNUP, userData),
  getProfile: async () => axiosInstance.get(AUTH_ROUTES.PROFILE),
  updateProfile: async (profileData) =>
    axiosInstance.put(AUTH_ROUTES.PROFILE, profileData),
  updatePassword: async (passwordData) =>
    axiosInstance.put(AUTH_ROUTES.UPDATE_PASSWORD, passwordData),
  // IMPORTANT: Make an actual API call for logout
  logout: async () => axiosInstance.post(AUTH_ROUTES.LOGOUT),
};
