// src/middleware/error.middleware.js

import { ApiError } from "../utils/ApiError.js";

const errorMiddleware = (err, req, res, next) => {
    if (err instanceof ApiError) {
        return res.status(err.statusCode).json({
            success: false,
            message: err.message,
            errors: err.errors,
        });
    }

    console.error("UNHANDLED_ERROR: ", err);

    return res.status(500).json({
        success: false,
        message: "Internal Server Error",
    });
};

export { errorMiddleware };
