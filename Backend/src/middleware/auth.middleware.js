// src/middleware/auth.middleware.js

import { ApiError } from "../utils/ApiError.js";
import { asyncHandler } from "../utils/asyncHandler.js";
import { verifyToken } from "../utils/jwt.js";
import { userRepository } from "../repositories/user.repository.js";

export const authenticateToken = asyncHandler(async (req, res, next) => {
    const token =
        req.cookies?.accessToken ||
        req.header("Authorization")?.replace("Bearer ", "");

    if (!token) {
        throw new ApiError(401, "Access token is required");
    }

    try {
        const decodedPayload = verifyToken(token);
        const user = await userRepository.findById(decodedPayload?.userId);

        if (!user || !user.isActive) {
            throw new ApiError(401, "Invalid access token or user not found");
        }

        req.user = user;
        next();
    } catch (error) {
        throw new ApiError(401, error?.message || "Invalid or expired token");
    }
});
