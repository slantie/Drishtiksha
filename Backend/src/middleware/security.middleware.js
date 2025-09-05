// src/middleware/security.middleware.js

import rateLimit from 'express-rate-limit';
import { ApiError } from '../utils/ApiError.js';

const createRateLimiter = (options) => {
    return rateLimit({
        windowMs: 15 * 60 * 1000,
        legacyHeaders: false,
        standardHeaders: true,
        handler: (req, res, next, options) => {
            throw new ApiError(
                options.statusCode,
                `Too many requests. Please try again after ${Math.ceil(options.windowMs / 60000)} minutes.`
            );
        },
        ...options,
    });
};

// A strict rate limiter for failed login attempts to prevent brute-force attacks.
export const loginRateLimiter = createRateLimiter({
    windowMs: 10 * 60 * 1000,
    max: 10,
    message: 'Too many failed login attempts. Please try again after 10 minutes.',
    skipSuccessfulRequests: true, // Only count failed requests
});

// A general rate limiter for all other API requests.
export const apiRateLimiter = createRateLimiter({
    windowMs: 15 * 60 * 1000,
    max: 200,
});