// src/api/monitoring/monitoring.routes.js

import express from "express";
import {
    getServerHealth,
    getServerHealthHistory,
    getAnalysisStats,
    getModelMetrics,
} from "./monitoring.controller.js";
// import { authenticateToken } from "../../middleware/auth.middleware.js";

const router = express.Router();

// Public health endpoint (for load balancers, monitoring services)
router.get("/health", getServerHealth);

// Monitoring endpoints (made public for easier testing and operational monitoring)
// In production, consider adding IP restrictions or API key authentication
router.get("/health/history", getServerHealthHistory);
router.get("/stats/analysis", getAnalysisStats);
router.get("/models/metrics", getModelMetrics);

export default router;
