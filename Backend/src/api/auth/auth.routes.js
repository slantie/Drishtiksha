// src/api/auth/auth.routes.js

import express from 'express';
import { authController } from './auth.controller.js';
import { validate, signupSchema, loginSchema, updateProfileSchema, updatePasswordSchema } from './auth.validation.js';
import { authenticateToken } from '../../middleware/auth.middleware.js';
import { loginRateLimiter } from '../../middleware/security.middleware.js';

const router = express.Router();

router.post('/signup', validate(signupSchema), authController.signup);
router.post('/login', loginRateLimiter, validate(loginSchema), authController.login);
router.post('/logout', authController.logout);

router.use(authenticateToken);

router.route('/profile')
    .get(authController.getProfile)
    .put(validate(updateProfileSchema), authController.updateProfile);

router.route('/profile/password')
    .put(validate(updatePasswordSchema), authController.updatePassword);

export default router;