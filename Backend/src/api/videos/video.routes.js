// src/api/videos/video.routes.js

import express from "express";
import { authenticateToken } from "../../middleware/auth.middleware.js";
import { upload } from "../../middleware/multer.middleware.js";
import { videoController } from "./video.controller.js";
import {
    validate,
    videoUpdateSchema,
    analysisRequestSchema,
} from "./video.validation.js";

const router = express.Router();

// --- PROTECTED ROUTES ---
// The authentication middleware is applied here.
// All routes defined BELOW this line will require a valid JWT.
router.use(authenticateToken);

// Model server status route - now protected
router.route("/status").get(videoController.getModelServerStatus);

// Available models route - get models that are currently active
router.route("/models").get(videoController.getAvailableModels);

// POST /api/v1/videos -> Upload a new video for analysis.
// GET /api/v1/videos  -> Get a list of all videos for the authenticated user.
router
    .route("/")
    .post(upload.single("video"), videoController.uploadVideo)
    .get(videoController.getAllVideos);

// GET /api/v1/videos/:id    -> Get a specific video with all its analysis results.
// DELETE /api/v1/videos/:id -> Delete a video and its associated data.
router
    .route("/:id")
    .get(videoController.getVideoById)
    .delete(videoController.deleteVideo);

export default router;
