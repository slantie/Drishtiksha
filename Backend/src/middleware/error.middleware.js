// src/middleware/error.middleware.js

import { ApiError } from '../utils/ApiError.js';
import logger from '../utils/logger.js';
import { config } from '../config/env.js';

export const errorMiddleware = (err, req, res, next) => {
    // If the error is an instance of our custom ApiError, we trust its properties.
    if (err instanceof ApiError) {
        return res.status(err.statusCode).json({
            success: false,
            message: err.message,
            errors: err.errors,
            // Optionally include stack trace in development for easier debugging.
            stack: config.NODE_ENV === 'development' ? err.stack : undefined,
        });
    }

    // For all other unexpected errors, log it as a critical error.
    logger.error(`UNHANDLED_ERROR: ${err.message}`, {
        stack: err.stack,
        url: req.originalUrl,
        method: req.method,
        ip: req.ip,
    });

    // And send a generic, safe response to the client.
    return res.status(500).json({
        success: false,
        message: 'An unexpected internal server error occurred.',
    });
};