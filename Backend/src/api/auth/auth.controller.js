// src/api/auth/auth.controller.js

import { authService } from '../../services/auth.service.js';
import { asyncHandler } from '../../utils/asyncHandler.js';
import { ApiResponse } from '../../utils/ApiResponse.js';
import logger from '../../utils/logger.js';

const signup = asyncHandler(async (req, res) => {
    const { user, token } = await authService.registerUser(req.body);
    logger.info(`[Auth] New user registered: ${user.email} (ID: ${user.id})`);
    res.status(201).json(new ApiResponse(201, { user, token }, 'User created successfully'));
});

const login = asyncHandler(async (req, res) => {
    const { email, password } = req.body;
    const { user, token } = await authService.loginUser(email, password);
    logger.info(`[Auth] User logged in: ${user.email} (ID: ${user.id})`);
    res.status(200).json(new ApiResponse(200, { user, token }, 'Login successful'));
});

const logout = asyncHandler(async (req, res) => {
    if (req.user) {
        logger.info(`[Auth] User logged out: ${req.user.email} (ID: ${req.user.id})`);
    }
    res.status(200).json(new ApiResponse(200, null, 'Logout successful'));
});

const getProfile = asyncHandler(async (req, res) => {
    const user = await authService.getUserProfile(req.user.id);
    res.status(200).json(new ApiResponse(200, user, 'Profile retrieved successfully'));
});

const updateProfile = asyncHandler(async (req, res) => {
    const updatedUser = await authService.updateUserProfile(req.user.id, req.body);
    logger.info(`[Auth] User profile updated: ${updatedUser.email} (ID: ${updatedUser.id})`);
    res.status(200).json(new ApiResponse(200, updatedUser, 'Profile updated successfully'));
});

const updatePassword = asyncHandler(async (req, res) => {
    const { currentPassword, newPassword } = req.body;
    await authService.changePassword(req.user.id, currentPassword, newPassword);
    logger.info(`[Auth] User password updated for: ${req.user.email} (ID: ${req.user.id})`);
    res.status(200).json(new ApiResponse(200, null, 'Password updated successfully'));
});

export const authController = {
    signup,
    login,
    logout,
    getProfile,
    updateProfile,
    updatePassword,
};