// src/middleware/multer.middleware.js

import multer from "multer";
import path from "path";
import fs from "fs";
import { ApiError } from "../utils/ApiError.js";

const TEMP_UPLOAD_DIR = path.resolve("temp/uploads");

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

// Define allowed MIME types for uploads
const ALLOWED_MIME_TYPES = new Set([
  // Video formats
  "video/mp4",
  "video/webm",
  "video/quicktime",
  "video/x-msvideo",
  "video/avi",
  "video/x-matroska", // MKV files
  
  // Image formats
  "image/jpeg",
  "image/jpg",
  "image/png",
  "image/webp",
  "image/gif",
  
  // Audio formats
  "audio/mpeg",
  "audio/mp3",
  "audio/wav",
  "audio/ogg",
  "audio/mp4",
  "audio/x-m4a",
  
  // Some files report as application/mp4
  "application/mp4",
]);

export const upload = multer({
  storage: storage,

  fileFilter: (req, file, cb) => {
    // Normalize certain MIME types that may vary
    if (file.mimetype === "application/mp4") {
      file.mimetype = "video/mp4";
    }
    
    // Validate file extension matches MIME type to prevent spoofing
    const fileExtension = path.extname(file.originalname).toLowerCase();
    const mimeType = file.mimetype.toLowerCase();
    
    // Basic extension validation
    const validExtensions = {
      'video/mp4': ['.mp4', '.m4v'],
      'video/webm': ['.webm'],
      'video/quicktime': ['.mov'],
      'video/x-msvideo': ['.avi'],
      'video/avi': ['.avi'],
      'video/x-matroska': ['.mkv'],
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/jpg': ['.jpg', '.jpeg'],
      'image/png': ['.png'],
      'image/webp': ['.webp'],
      'image/gif': ['.gif'],
      'audio/mpeg': ['.mp3', '.mpeg'],
      'audio/mp3': ['.mp3'],
      'audio/wav': ['.wav'],
      'audio/ogg': ['.ogg'],
      'audio/mp4': ['.m4a'],
      'audio/x-m4a': ['.m4a'],
    };
    
    if (ALLOWED_MIME_TYPES.has(mimeType)) {
      const allowedExts = validExtensions[mimeType] || [];
      if (allowedExts.length === 0 || allowedExts.includes(fileExtension)) {
        cb(null, true);
      } else {
        const err = new ApiError(
          415,
          `File extension '${fileExtension}' does not match MIME type '${mimeType}'.`
        );
        cb(err, false);
      }
    } else {
      const err = new ApiError(
        415,
        `Unsupported file type: '${file.mimetype}'. Please upload a valid video, image, or audio file.`
      );
      cb(err, false);
    }
  },
  limits: { fileSize: 150 * 1024 * 1024 }, // 150MB limit
});
