// src/app.js

import express from "express";
import cors from "cors";
import cookieParser from "cookie-parser";
import { errorMiddleware } from "./middleware/error.middleware.js";

const app = express();

// --- Core Middleware ---
app.use(
    cors({
        origin: process.env.FRONTEND_URL || "*",
        credentials: true,
    })
);

app.use(express.json({ limit: "16kb" }));
app.use(express.urlencoded({ extended: true, limit: "16kb" }));
app.use(express.static("public"));
app.use(cookieParser());

// --- Route Imports ---
import authRoutes from "./api/auth/auth.routes.js";
import videoRoutes from "./api/videos/video.routes.js";
import healthRoutes from "./api/health/health.routes.js";
import monitoringRoutes from "./api/monitoring/monitoring.routes.js";

// --- API Routes ---
// It's good practice to version your API.
app.use("/health", healthRoutes);
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
// This must be the LAST middleware added.
app.use(errorMiddleware);

export { app };
