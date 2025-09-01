// src/middleware/multer.middleware.js

import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { ApiError } from '../utils/ApiError.js';

const TEMP_UPLOAD_DIR = path.resolve('temp/uploads');

// Ensure the temporary directory exists at startup.
fs.mkdirSync(TEMP_UPLOAD_DIR, { recursive: true });

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, TEMP_UPLOAD_DIR);
    },
    filename: (req, file, cb) => {
        const uniqueSuffix = `${Date.now()}-${Math.round(Math.random() * 1e9)}`;
        const extension = path.extname(file.originalname);
        cb(null, `upload-${uniqueSuffix}${extension}`);
    },
});

const ALLOWED_MIME_TYPES = new Set([
    'video/mp4', 'video/webm', 'video/quicktime', 'video/x-msvideo',
    'image/jpeg', 'image/png', 'image/webp',
    'audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/mp4',
]);

export const upload = multer({
    storage: storage,
    fileFilter: (req, file, cb) => {
        if (ALLOWED_MIME_TYPES.has(file.mimetype)) {
            cb(null, true);
        } else {
            const err = new ApiError(415, `Unsupported file type: '${file.mimetype}'.`);
            cb(err, false);
        }
    },
    limits: { fileSize: 150 * 1024 * 1024 }, // 150MB limit
});