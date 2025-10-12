// src/api/pdf/pdf.routes.js

import express from 'express';
import { authenticateToken } from '../../middleware/auth.middleware.js';
import { pdfController } from './pdf.controller.js';
import { validate, analysisRunIdParamSchema } from './pdf.validation.js';

const router = express.Router();

// Apply authentication middleware
router.use(authenticateToken);

// GET /api/v1/pdf/report/run/:analysisRunId - Generate and download PDF report for a specific analysis run
router.get('/report/run/:analysisRunId', validate(analysisRunIdParamSchema), pdfController.generateRunReport);

// GET /api/v1/pdf/test - Generate a simple test PDF (for debugging)
router.get('/test', pdfController.generateTestPDF);

export default router;
