// src/middleware/auth.middleware.js

import { ApiError } from '../utils/ApiError.js';
import { asyncHandler } from '../utils/asyncHandler.js';
import { verifyToken } from '../utils/jwt.js';
import { userRepository } from '../repositories/user.repository.js';

// This is a placeholder for a central constants file if we need one later.
const AUTH_COOKIE_NAME = 'accessToken'; 

export const authenticateToken = asyncHandler(async (req, res, next) => {
    // Check for token in standard Authorization header first, then fallback to cookie.
    const token = req.header('Authorization')?.replace('Bearer ', '') || req.cookies?.[AUTH_COOKIE_NAME];

    if (!token) {
        throw new ApiError(401, 'Unauthorized: Access token is required.');
    }

    try {
        const decodedPayload = verifyToken(token);
        if (!decodedPayload?.userId) {
             throw new ApiError(401, 'Invalid token payload.');
        }

        const user = await userRepository.findById(decodedPayload.userId);

        if (!user || !user.isActive) {
            throw new ApiError(401, 'Invalid access token or user is inactive.');
        }

        req.user = user;
        next();
    } catch (error) {
        // Catch JWT errors (like expiry) and standardize the response.
        throw new ApiError(401, error?.message || 'Invalid or expired token.');
    }
});