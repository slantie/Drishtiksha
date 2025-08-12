// src/api/auth/auth.routes.js

import express from "express";
import { authController } from "./auth.controller.js";
import {
    validate,
    signupSchema,
    loginSchema,
    updateProfileSchema,
    updatePasswordSchema,
    updateAvatarSchema,
} from "./auth.validation.js";
import { authenticateToken } from "../../middleware/auth.middleware.js";

const router = express.Router();

// Public routes
router.route("/signup").post(validate(signupSchema), authController.signup);
router.route("/login").post(validate(loginSchema), authController.login);
router.route("/logout").post(authController.logout);

// Protected routes - All routes below this will use the authenticateToken middleware
router.use(authenticateToken);

router.route("/profile").get(authController.getProfile);
router
    .route("/profile")
    .put(validate(updateProfileSchema), authController.updateProfile);
router
    .route("/profile/password")
    .put(validate(updatePasswordSchema), authController.updatePassword);
router
    .route("/profile/avatar")
    .put(validate(updateAvatarSchema), authController.updateAvatar);
router.route("/profile/avatar").delete(authController.deleteAvatar);

export default router;
