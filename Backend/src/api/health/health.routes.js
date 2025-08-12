// src/api/health/health.routes.js

import express from "express";
import prisma from "../../config/database.js";

const router = express.Router();

router.get("/", async (req, res) => {
    try {
        await prisma.user.findFirst();

        res.json({
            success: true,
            status: "OK",
            timestamp: new Date().toISOString(),
            database: "Connected",
            environment: process.env.NODE_ENV,
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
