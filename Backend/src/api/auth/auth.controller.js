// src/api/auth/auth.controller.js

import { authService } from "../../services/auth.service.js";
import { asyncHandler } from "../../utils/asyncHandler.js";
import { ApiResponse } from "../../utils/ApiResponse.js";

const signup = asyncHandler(async (req, res) => {
    const { user, token } = await authService.registerUser(req.body);
    const response = new ApiResponse(
        201,
        { user, token },
        "User created successfully"
    );
    res.status(response.statusCode).json(response);
});

const login = asyncHandler(async (req, res) => {
    const { email, password } = req.body;
    const { user, token } = await authService.loginUser(email, password);
    const response = new ApiResponse(200, { user, token }, "Login successful");
    res.status(response.statusCode).json(response);
});

const logout = asyncHandler(async (req, res) => {
    const response = new ApiResponse(200, null, "Logout successful");
    res.status(response.statusCode).json(response);
});

const getProfile = asyncHandler(async (req, res) => {
    const user = await authService.getUserProfile(req.user.id);
    const response = new ApiResponse(
        200,
        user,
        "Profile retrieved successfully"
    );
    res.status(response.statusCode).json(response);
});

const updateProfile = asyncHandler(async (req, res) => {
    const updatedUser = await authService.updateUserProfile(
        req.user.id,
        req.body
    );
    const response = new ApiResponse(
        200,
        updatedUser,
        "Profile updated successfully"
    );
    res.status(response.statusCode).json(response);
});

const updatePassword = asyncHandler(async (req, res) => {
    const { currentPassword, newPassword } = req.body;
    await authService.changePassword(req.user.id, currentPassword, newPassword);
    const response = new ApiResponse(
        200,
        null,
        "Password updated successfully"
    );
    res.status(response.statusCode).json(response);
});

const updateAvatar = asyncHandler(async (req, res) => {
    const updatedUser = await authService.updateUserAvatar(
        req.user.id,
        req.body.avatar
    );
    const response = new ApiResponse(
        200,
        updatedUser,
        "Avatar updated successfully"
    );
    res.status(response.statusCode).json(response);
});

const deleteAvatar = asyncHandler(async (req, res) => {
    const updatedUser = await authService.removeUserAvatar(req.user.id);
    const response = new ApiResponse(
        200,
        updatedUser,
        "Avatar removed successfully"
    );
    res.status(response.statusCode).json(response);
});

export const authController = {
    signup,
    login,
    logout,
    getProfile,
    updateProfile,
    updatePassword,
    updateAvatar,
    deleteAvatar,
};
