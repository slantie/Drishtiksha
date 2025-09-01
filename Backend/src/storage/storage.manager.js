// src/storage/storage.manager.js

import { config } from '../config/env.js';
import logger from '../utils/logger.js';

let storageManager;

if (config.STORAGE_PROVIDER === 'local') {
    const { default: localProvider } = await import('./local.provider.js');
    storageManager = localProvider;
    logger.info("📦 Storage Manager initialized with 'local' filesystem provider.");
} else {
    const { default: cloudinaryProvider } = await import('./cloudinary.provider.js');
    storageManager = cloudinaryProvider;
    logger.info("☁️ Storage Manager initialized with 'cloudinary' provider.");
}

export default storageManager;