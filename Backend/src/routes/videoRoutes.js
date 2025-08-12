/**
 * @file src/routes/videoRoutes.js
 * @description API routes for video management, including upload, analysis, and CRUD operations.
 * This file has been refactored to use dedicated controller functions for better separation of concerns.
 */

import express from "express";
import multer from "multer";
import { authenticateToken } from "../middleware/auth.js";
import {
    processVideoForDeepfake,
    getAnalysisResults,
    getAllAnalyses,
} from "../controllers/deepfakeController.js";
import {
    uploadVideo,
    getAllVideos,
    getVideoById,
    updateVideo,
    deleteVideo
} from "../controllers/videoController.js";
import path from "path";
import { PrismaClient } from "@prisma/client";

const router = express.Router();
const prisma = new PrismaClient();

// Configure multer storage for video uploads
const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, "uploads/videos/"),
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
        cb(null, file.fieldname + "-" + uniqueSuffix + path.extname(file.originalname));
    },
});

// Configure multer with file type and size limits
const upload = multer({
    storage: storage,
    fileFilter: (req, file, cb) => {
        if (file.mimetype.startsWith("video/")) {
            cb(null, true);
        } else {
            cb(new Error("Only video files are allowed!"), false);
        }
    },
    limits: { fileSize: 100 * 1024 * 1024 }, // 100MB
});

// --- Authenticated Routes ---
// All routes below this point require a valid JWT token.

// Route to upload a single video file
router.post("/upload", authenticateToken, upload.single("video"), uploadVideo);

// Route to analyze a video with a specific model
router.post("/:videoId/analyze", authenticateToken, processVideoForDeepfake);

// Get a video's specific analysis results (less used now, but kept for completeness)
router.get("/:videoId/analysis", authenticateToken, getAnalysisResults);

// Get all video analyses for the dashboard
router.get("/analyses", authenticateToken, getAllAnalyses);

// Get all of the current user's videos (or all if user is an ADMIN)
router.get("/", authenticateToken, getAllVideos);

// Get a single video by its ID
router.get("/:id", authenticateToken, getVideoById);

// Update a video's description
router.patch("/:id", authenticateToken, updateVideo);

// Delete a video and its file
router.delete("/:id", authenticateToken, deleteVideo);

export default router;
