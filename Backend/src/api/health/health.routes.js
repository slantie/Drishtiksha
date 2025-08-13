// src/api/health/health.routes.js

import express from "express";
import prisma from "../../config/database.js";
import { modelAnalysisService } from "../../services/modelAnalysis.service.js";

const router = express.Router();

router.get("/", async (req, res) => {
    try {
        await prisma.user.findFirst();

        // Check model service status
        let modelServiceStatus = {
            configured: modelAnalysisService.isAvailable(),
            status: "unknown",
            error: null,
        };

        if (modelServiceStatus.configured) {
            try {
                const healthCheck = await modelAnalysisService.checkHealth();
                modelServiceStatus.status =
                    healthCheck.status === "healthy"
                        ? "connected"
                        : "unhealthy";
                modelServiceStatus.modelLoaded = healthCheck.model_loaded;
                modelServiceStatus.defaultModel = healthCheck.default_model;
            } catch (error) {
                modelServiceStatus.status = "disconnected";
                modelServiceStatus.error = error.message;
            }
        } else {
            modelServiceStatus.status = "not_configured";
        }

        res.json({
            success: true,
            status: "OK",
            timestamp: new Date().toISOString(),
            database: "Connected",
            environment: process.env.NODE_ENV,
            modelService: modelServiceStatus,
        });
    } catch (error) {
        res.status(503).json({
            success: false,
            status: "ERROR",
            database: "Disconnected",
            error: error.message,
        });
    }
});

export default router;
