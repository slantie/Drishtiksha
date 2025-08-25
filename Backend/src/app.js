// src/app.js

import express from "express";
import cors from "cors";
import cookieParser from "cookie-parser";
import helmet from "helmet";
import rateLimit from "express-rate-limit";
import { errorMiddleware } from "./middleware/error.middleware.js";
// ADDED: These are required for path resolution
import { fileURLToPath } from "url";
import path from "path";
import logger from "./utils/logger.js";

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
app.use(cookieParser());

// --- SIMPLIFIED: Static File Serving for Local Storage ---
// This block is now cleaner and more direct.
if (process.env.STORAGE_PROVIDER === "local") {
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = path.dirname(__filename);

    // Get the root storage path from environment variables.
    const localStoragePath = process.env.LOCAL_STORAGE_PATH || "public/media";

    // Resolve the absolute path to the directory on the server's filesystem.
    const staticDirPath = path.resolve(__dirname, "..", localStoragePath);

    // Create a URL-safe path for Express to mount.
    // This logic correctly handles the base path for serving files.
    const staticUrlPath = localStoragePath
        .replace(/^public\//, "") // Remove 'public/' prefix to prevent it from being in the URL.
        .replace(/\\/g, "/"); // Normalize backslashes to forward slashes for URL compatibility.

    logger.info(
        `🚀 Serving static files from URL '/${staticUrlPath}' mapped to directory '${staticDirPath}'`
    );

    // Mount the static directory. Express handles the rest.
    app.use(`/${staticUrlPath}`, express.static(staticDirPath));
}

// --- Route Imports ---
import authRoutes from "./api/auth/auth.routes.js";
import mediaRoutes from "./api/media/media.routes.js";
import monitoringRoutes from "./api/monitoring/monitoring.routes.js";

// --- API Routes ---
app.use("/api/v1/auth", authRoutes);
app.use("/api/v1/media", mediaRoutes);
app.use("/api/v1/monitoring", monitoringRoutes);

app.get("/", (req, res) => {
    res.status(200).json({
        success: true,
        message: "Drishtiksha API is alive and running!",
    });
});

// --- Central Error Handling Middleware ---
app.use(errorMiddleware);

export { app };
