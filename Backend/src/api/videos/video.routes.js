// src/api/videos/video.routes.js

import express from "express";
import { authenticateToken } from "../../middleware/auth.middleware.js";
import { upload } from "../../middleware/multer.middleware.js";
import { videoController } from "./video.controller.js";
import { validate, videoUpdateSchema } from "./video.validation.js";

const router = express.Router();

// All routes in this file are protected and require authentication
router.use(authenticateToken);

router
    .route("/")
    .get(videoController.getAllVideos)
    .post(upload.single("video"), videoController.uploadVideo);

// Model service status endpoint
router.get("/model/status", videoController.getModelStatus);

router
    .route("/:id")
    .get(videoController.getVideoById)
    .patch(validate(videoUpdateSchema), videoController.updateVideo)
    .delete(videoController.deleteVideo);

export default router;
