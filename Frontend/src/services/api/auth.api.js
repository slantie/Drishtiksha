// src/services/api/auth.api.js

import axiosInstance from '../../lib/axios.js';

const AUTH_ROUTES = {
    LOGIN: '/auth/login',
    SIGNUP: '/auth/signup',
    LOGOUT: '/auth/logout',
    PROFILE: '/auth/profile',
    UPDATE_PASSWORD: '/auth/profile/password',
};

export const authApi = {
    login: async (credentials) => axiosInstance.post(AUTH_ROUTES.LOGIN, credentials),
    signup: async (userData) => axiosInstance.post(AUTH_ROUTES.SIGNUP, userData),
    getProfile: async () => axiosInstance.get(AUTH_ROUTES.PROFILE),
    updateProfile: async (profileData) => axiosInstance.put(AUTH_ROUTES.PROFILE, profileData),
    updatePassword: async (passwordData) => axiosInstance.put(AUTH_ROUTES.UPDATE_PASSWORD, passwordData),
    logout: async () => Promise.resolve(),
};