// src/app.js

import express from "express";
import cors from "cors";
import cookieParser from "cookie-parser";
import helmet from "helmet"; // ADDED: For security headers
import rateLimit from "express-rate-limit"; // ADDED: For rate limiting
import { errorMiddleware } from "./middleware/error.middleware.js";

const app = express();

// --- Core Middleware ---

// ADDED: Set security-related HTTP response headers
app.use(helmet());

app.use(
    cors({
        origin: process.env.FRONTEND_URL || "*",
        credentials: true,
    })
);

// ADDED: Basic rate limiting to prevent abuse
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 200, // Limit each IP to 200 requests per windowMs
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

// --- Route Imports ---
import authRoutes from "./api/auth/auth.routes.js";
import videoRoutes from "./api/videos/video.routes.js";
import healthRoutes from "./api/health/health.routes.js";
// ADDED: New routes for application monitoring
import monitoringRoutes from "./api/monitoring/monitoring.routes.js";

// --- API Routes ---
// It's good practice to version your API.
app.use("/health", healthRoutes); // Simple, un-versioned health check
app.use("/api/v1/auth", authRoutes);
app.use("/api/v1/videos", videoRoutes);
// ADDED: Register the new monitoring routes
app.use("/api/v1/monitoring", monitoringRoutes);

app.get("/", (req, res) => {
    res.status(200).json({
        success: true,
        message: "API is alive and running!",
    });
});

// --- Central Error Handling Middleware ---
// This must be the LAST middleware added.
app.use(errorMiddleware);

export { app };
