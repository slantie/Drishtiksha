// src/api/monitoring/monitoring.routes.js

import express from "express";
import { monitoringController } from "./monitoring.controller.js";
import { authenticateToken } from "../../middleware/auth.middleware.js";

const router = express.Router();

// Enforce authentication for all monitoring routes to protect internal system data.
router.use(authenticateToken);

// Route to get the live status of the ML server.
router.get("/server-status", monitoringController.getServerStatus);

// Route to get the history of ML server health checks.
router.get("/server-history", monitoringController.getServerHealthHistory);

// Route to get aggregated analysis performance statistics.
router.get("/stats/analysis", monitoringController.getAnalysisStats);

// Route to get the status of the video processing queue.
router.get("/queue-status", monitoringController.getQueueStatus);

// REASON: The '/models/metrics' route is removed as it's now redundant.
// Real-time data comes from '/server-status' and historical data from '/stats/analysis'.

export default router;
