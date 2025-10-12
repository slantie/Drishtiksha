// src/services/MediaDownloadService.js
// Service for downloading media files and PDF reports

import { showToast } from "../utils/toast.jsx";

/**
 * Service for downloading media files
 * Extracted from old DownloadReport.js to work with new PDFGenerator
 */
export const MediaDownloadService = {
  /**
   * Downloads the original media file
   * @param {string} mediaUrl - URL of the media file
   * @param {string} filename - Original filename
   */
  async downloadMedia(mediaUrl, filename) {
    try {
      if (!mediaUrl) {
        showToast.error("Media URL is missing for download.");
        throw new Error("Media URL is missing.");
      }

      // Assume Cloudinary URL based on file structure
      const url = new URL(mediaUrl);
      const parts = url.pathname.split("/upload/");

      if (parts.length > 1) {
        // Ensure we get the full public ID which includes subfolders and original filename base
        const publicIdPath = parts[1].split("/").slice(1).join("/"); // Remove 'f_auto,q_auto' or similar
        const baseCloudinaryPath = parts[0];

        // Construct the new download URL with fl_attachment and f_auto (for optimal format conversion)
        const newPathname = `${baseCloudinaryPath}/upload/fl_attachment/${publicIdPath}`;
        url.pathname = newPathname;
      } else {
        // For non-Cloudinary or local paths, attempt direct download
        console.warn(
          "Non-Cloudinary URL structure detected for download. Attempting direct download."
        );
        // Simply use the provided URL if it's not a Cloudinary transformed URL
      }

      const link = document.createElement("a");
      link.href = url.href;
      link.download = filename || "download_file"; // Ensure filename is provided or default
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      showToast.success(`Download started: ${filename}`);
    } catch (error) {
      console.error("Error downloading media:", error);
      showToast.error("Failed to download media. Please try again.");
      throw error;
    }
  },
};
