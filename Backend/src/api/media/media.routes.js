// src/api/media/media.routes.js

import express from "express";
import { authenticateToken } from "../../middleware/auth.middleware.js";
import { upload } from "../../middleware/multer.middleware.js";
// RENAMED: Importing mediaController instead of videoController
import { mediaController } from "./media.controller.js";

const router = express.Router();

router.use(authenticateToken);

router
    .route("/")
    // UPDATED: The field name for multer is now "file" to be generic.
    .post(upload.single("file"), mediaController.uploadMedia)
    .get(mediaController.getAllMedia);

router
    .route("/:id")
    .get(mediaController.getMediaById)
    .patch(mediaController.updateMedia)
    .delete(mediaController.deleteMedia);

export default router;
