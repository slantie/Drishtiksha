// src/api/media/media.routes.js

import express from 'express';
import { authenticateToken } from '../../middleware/auth.middleware.js';
import { upload } from '../../middleware/multer.middleware.js';
import { mediaController } from './media.controller.js';
import { validate, mediaUpdateSchema, mediaIdParamSchema } from './media.validation.js';

const router = express.Router();

router.use(authenticateToken);

router.route('/')
    .post(upload.single('media'), mediaController.uploadMedia)
    .get(mediaController.getAllMedia);

router.route('/:id')
    .get(validate(mediaIdParamSchema), mediaController.getMediaById)
    .patch(validate(mediaUpdateSchema), mediaController.updateMedia)
    .delete(validate(mediaIdParamSchema), mediaController.deleteMedia);

router.route('/:id/analyze')
    .post(validate(mediaIdParamSchema), mediaController.rerunAnalysis);

export default router;