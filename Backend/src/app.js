// src/app.js

import express from "express";
import cors from "cors";
import cookieParser from "cookie-parser";
import helmet from "helmet";
import rateLimit from "express-rate-limit";
import { errorMiddleware } from "./middleware/error.middleware.js";

const app = express();

// --- Core Middleware ---
app.use(
    helmet({
        crossOriginResourcePolicy: { policy: "cross-origin" },
    })
);
app.use(
    cors({
        origin: process.env.FRONTEND_URL || "*",
        credentials: true,
        methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allowedHeaders: [
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-API-Key",
        ],
        exposedHeaders: ["Content-Disposition", "Content-Length"],
        preflightContinue: false,
        optionsSuccessStatus: 204,
    })
);

// Handle CORS preflight for all routes
app.options("*", cors());

const limiter = rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 200,
    standardHeaders: true,
    legacyHeaders: false,
    message:
        "Too many requests from this IP, please try again after 15 minutes",
});
app.use(limiter);

app.use(express.json({ limit: "16kb" }));
app.use(express.urlencoded({ extended: true, limit: "16kb" }));
app.use(express.static("public"));
app.use(cookieParser());

if (process.env.STORAGE_PROVIDER === "local") {
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = path.dirname(__filename);

    // Get the configured path, e.g., "database/public/media"
    const localStoragePath = process.env.LOCAL_STORAGE_PATH || "public/media";

    // Create a safe URL path for static serving
    let staticUrlPath = localStoragePath
        .replace(/\\/g, "/") // Convert Windows backslashes to forward slashes
        .replace(/^public\//, "") // Remove leading public/
        .replace(/^\/+|\/+$/g, "") // Remove leading and trailing slashes
        .replace(/\/+/g, "/") // Replace multiple slashes with single slash
        .replace(/[^a-zA-Z0-9\-_\/]/g, ""); // Remove special characters that might break path-to-regexp

    // Fallback to a safe default if the path is empty or invalid
    if (!staticUrlPath || staticUrlPath === "/") {
        staticUrlPath = "media";
    }

    // Ensure it doesn't start with a slash (Express will add it)
    staticUrlPath = staticUrlPath.replace(/^\/+/, "");

    const staticDirPath = path.resolve(__dirname, "..", localStoragePath);

    // Log the paths for debugging
    console.log("Local storage path:", localStoragePath);
    console.log("Static URL path:", staticUrlPath);
    console.log("Static directory path:", staticDirPath);

    try {
        app.use(
            `/${staticUrlPath}`,
            (req, res, next) => {
                // Set CORS headers for all requests
                res.header(
                    "Access-Control-Allow-Origin",
                    process.env.FRONTEND_URL || "*"
                );
                res.header("Access-Control-Allow-Methods", "GET,OPTIONS");
                res.header(
                    "Access-Control-Allow-Headers",
                    "Origin, X-Requested-With, Content-Type, Accept, Authorization"
                );
                res.header("Access-Control-Allow-Credentials", "true");

                // Handle preflight OPTIONS requests
                if (req.method === "OPTIONS") {
                    res.status(200).end();
                    return;
                }

                next();
            },
            express.static(staticDirPath)
        );

        logger.info(
            `🚀 Serving static files for local storage from URL '/${staticUrlPath}' mapped to directory '${staticDirPath}'`
        );
    } catch (error) {
        console.error("Error setting up static file serving:", error);
        logger.error("Failed to set up static file serving:", error);

        // Fallback to a simple media route
        app.use(
            "/media",
            (req, res, next) => {
                // Set CORS headers for all requests
                res.header(
                    "Access-Control-Allow-Origin",
                    process.env.FRONTEND_URL || "*"
                );
                res.header("Access-Control-Allow-Methods", "GET,OPTIONS");
                res.header(
                    "Access-Control-Allow-Headers",
                    "Origin, X-Requested-With, Content-Type, Accept, Authorization"
                );
                res.header("Access-Control-Allow-Credentials", "true");

                // Handle preflight OPTIONS requests
                if (req.method === "OPTIONS") {
                    res.status(200).end();
                    return;
                }

                next();
            },
            express.static(staticDirPath)
        );

        logger.info(
            `🚀 Fallback: Serving static files from URL '/media' mapped to directory '${staticDirPath}'`
        );
    }
}

// --- Route Imports ---
import authRoutes from "./api/auth/auth.routes.js";
import videoRoutes from "./api/videos/video.routes.js";
import monitoringRoutes from "./api/monitoring/monitoring.routes.js";
import { fileURLToPath } from "url";
import path from "path";
import logger from "./utils/logger.js";
// REMOVED: import healthRoutes from "./api/health/health.routes.js";

// --- API Routes ---
// REMOVED: app.use("/health", healthRoutes);
app.use("/api/v1/auth", authRoutes);
app.use("/api/v1/videos", videoRoutes);
app.use("/api/v1/monitoring", monitoringRoutes);

app.get("/", (req, res) => {
    res.status(200).json({
        success: true,
        message: "API is alive and running!",
    });
});

// --- Central Error Handling Middleware ---
app.use(errorMiddleware);

export { app };
