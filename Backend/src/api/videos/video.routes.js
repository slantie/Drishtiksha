// src/api/videos/video.routes.js

import express from "express";
import { authenticateToken } from "../../middleware/auth.middleware.js";
import { upload } from "../../middleware/multer.middleware.js";
import { videoController } from "./video.controller.js";

const router = express.Router();

router.use(authenticateToken);

// REMOVED: Obsolete routes. This functionality now lives in the monitoring module.
// router.route("/status").get(videoController.getModelServerStatus);
// router.route("/models").get(videoController.getAvailableModels);

router
    .route("/")
    .post(upload.single("video"), videoController.uploadVideo)
    .get(videoController.getAllVideos);

router
    .route("/:id")
    .get(videoController.getVideoById)
    .patch(videoController.updateVideo)
    .delete(videoController.deleteVideo);

export default router;
