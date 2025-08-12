// src/middleware/multer.middleware.js

import multer from "multer";
import path from "path";

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        // The destination folder for temporary local storage
        cb(null, "uploads/videos/");
    },
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
        cb(
            null,
            file.fieldname +
                "-" +
                uniqueSuffix +
                path.extname(file.originalname)
        );
    },
});

export const upload = multer({
    storage: storage,
    fileFilter: (req, file, cb) => {
        if (file.mimetype.startsWith("video/")) {
            cb(null, true);
        } else {
            cb(
                new Error(
                    "File type not supported. Only video files are allowed."
                ),
                false
            );
        }
    },
    limits: { fileSize: 100 * 1024 * 1024 }, // 100MB limit
});
