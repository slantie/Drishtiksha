// src/utils/media.js

/**
 * @fileoverview Utility functions for handling different media types.
 */

/**
 * Determines the application's internal MediaType enum from a file's MIME type.
 * This is crucial for dispatching the correct logic for video, audio, or images.
 *
 * @param {string} mimetype - The MIME type of the file (e.g., "video/mp4", "image/jpeg").
 * @returns {'VIDEO' | 'IMAGE' | 'AUDIO' | null} The corresponding MediaType enum value, or null if unsupported.
 */
export const getMediaType = (mimetype) => {
    if (!mimetype) {
        return null;
    }

    if (mimetype.startsWith("video/")) {
        return "VIDEO";
    }
    if (mimetype.startsWith("image/")) {
        return "IMAGE";
    }
    if (mimetype.startsWith("audio/")) {
        return "AUDIO";
    }

    // Return null for any other type (e.g., "application/pdf")
    return null;
};
