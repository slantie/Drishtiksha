// src/middleware/multer.middleware.js

import multer from "multer";
import path from "path";
import fs from "fs";

// --- NEW: Define a temporary directory for all uploads ---
const TEMP_UPLOAD_DIR = "temp/uploads";

// Ensure the temporary directory exists
fs.mkdirSync(TEMP_UPLOAD_DIR, { recursive: true });

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        // All files go to a single, generic temporary directory
        cb(null, TEMP_UPLOAD_DIR);
    },
    filename: (req, file, cb) => {
        // Create a unique filename to prevent collisions
        const uniqueSuffix = `${Date.now()}-${Math.round(Math.random() * 1e9)}`;
        const extension = path.extname(file.originalname);
        cb(null, `${file.fieldname}-${uniqueSuffix}${extension}`);
    },
});

// --- NEW: Define allowed MIME types for all supported media ---
const ALLOWED_MIME_TYPES = [
    "video/mp4",
    "video/webm",
    "video/quicktime", // .mov
    "video/x-msvideo", // .avi
    "image/jpeg",
    "image/png",
    "image/webp",
    "audio/mpeg", // .mp3
    "audio/wav",
    "audio/ogg",
    "audio/mp4", // .m4a
];

export const upload = multer({
    storage: storage,
    fileFilter: (req, file, cb) => {
        // --- UPDATED: Check if the uploaded file's mimetype is in our allowed list ---
        if (ALLOWED_MIME_TYPES.includes(file.mimetype)) {
            cb(null, true);
        } else {
            // Provide a more informative error message
            const supportedTypes = ALLOWED_MIME_TYPES.map(
                (t) => t.split("/")[1]
            ).join(", ");
            cb(
                new Error(
                    `File type not supported. Allowed types: ${supportedTypes}`
                ),
                false
            );
        }
    },
    // Increased limit slightly for flexibility, e.g., high-res images or audio
    limits: { fileSize: 150 * 1024 * 1024 }, // 150MB limit
});
