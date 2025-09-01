// src/api/monitoring/monitoring.routes.js

import express from 'express';
import { monitoringController } from './monitoring.controller.js';
import { authenticateToken } from '../../middleware/auth.middleware.js';

const router = express.Router();

router.use(authenticateToken);

router.get('/server-status', monitoringController.getServerStatus);
router.get('/server-history', monitoringController.getServerHealthHistory);
router.get('/queue-status', monitoringController.getQueueStatus);

export default router;