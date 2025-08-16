// src/utils/formatters.js

/**
 * @fileoverview Centralized utility functions for formatting data across the application.
 */

/**
 * Formats bytes into a human-readable string (e.g., "1.23 MB").
 * @param {number} bytes - The number of bytes.
 * @returns {string} The formatted file size.
 */
export const formatBytes = (bytes) => {
    if (!bytes || bytes === 0) return "0 B";
    if (isNaN(bytes)) return "N/A";
    const sizes = ["B", "KB", "MB", "GB", "TB"];
    const i = parseInt(Math.floor(Math.log(bytes) / Math.log(1024)), 10);
    if (i === 0) return `${bytes} ${sizes[i]}`;
    return `${(bytes / 1024 ** i).toFixed(2)} ${sizes[i]}`;
};

/**
 * Formats an ISO date string into a more readable local date and time.
 * @param {string} dateString - The ISO date string to format.
 * @returns {string} The formatted date string.
 */
export const formatDate = (dateString) => {
    if (!dateString) return "N/A";
    try {
        return new Date(dateString).toLocaleString("en-US", {
            year: "numeric",
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
            hour12: true,
        });
    } catch (e) {
        return "Invalid Date";
    }
};

/**
 * Formats a duration in seconds into a human-readable string (e.g., "1m 23.4s").
 * @param {number} timeInSeconds - The duration in seconds.
 * @returns {string} The formatted duration.
 */
export const formatProcessingTime = (timeInSeconds) => {
    if (timeInSeconds === null || typeof timeInSeconds === "undefined")
        return "N/A";
    if (timeInSeconds < 60) return `${timeInSeconds.toFixed(1)}s`;
    const minutes = Math.floor(timeInSeconds / 60);
    const seconds = (timeInSeconds % 60).toFixed(1);
    return `${minutes}m ${seconds}s`;
};
