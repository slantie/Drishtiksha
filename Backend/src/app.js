// src/app.js

import express from "express";
import cors from "cors";
import cookieParser from "cookie-parser";
import helmet from "helmet";
import rateLimit from "express-rate-limit";
import { errorMiddleware } from "./middleware/error.middleware.js";

const app = express();

// --- Core Middleware ---
app.use(helmet());
app.use(cors({ origin: process.env.FRONTEND_URL || "*", credentials: true }));

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

// Storage Setup
if (process.env.STORAGE_PROVIDER === "local") {
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = path.dirname(__filename);
    
    // Get the configured path, e.g., "public/media"
    const localStoragePath = process.env.LOCAL_STORAGE_PATH || "public/media";
    
    // This creates the URL segment, e.g., "/media"
    const staticUrlPath = localStoragePath.replace("public/", "");

    // This creates the absolute filesystem path to the directory
    const staticDirPath = path.resolve(__dirname, '..', localStoragePath);

    // Serve files from the local storage directory at the specified URL
    app.use(`/${staticUrlPath}`, express.static(staticDirPath));
    
    logger.info(`🚀 Serving static files for local storage from URL '/${staticUrlPath}' mapped to directory '${staticDirPath}'`);
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
