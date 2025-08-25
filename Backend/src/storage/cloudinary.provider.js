// src/storage/cloudinary.provider.js

import {
    uploadOnCloudinary,
    uploadStreamToCloudinary,
    deleteFromCloudinary,
} from "../utils/cloudinary.js";
import { getMediaType } from "../utils/media.js";
import path from "path";

const cloudinaryProvider = {
    /**
     * Uploads a file from a local path to Cloudinary, automatically determining resource type.
     * @param {string} localFilePath - The temporary local path of the file.
     * @param {string} [subfolder] - Optional subfolder name, defaults to media type (e.g., 'videos').
     * @returns {Promise<{url: string, publicId: string}>} - The public URL and ID of the uploaded file.
     */
    async uploadFile(localFilePath, subfolder) {
        // --- NEW: Automatically determine resource_type based on file extension ---
        // This is a robust fallback for Cloudinary's auto-detection.
        const extension = path.extname(localFilePath).toLowerCase();
        let resource_type = "auto";
        if ([".mp4", ".mov", ".avi", ".webm"].includes(extension)) {
            resource_type = "video";
        } else if (
            [".jpg", ".jpeg", ".png", ".gif", ".webp"].includes(extension)
        ) {
            resource_type = "image";
        } else if ([".mp3", ".wav", ".ogg", ".m4a"].includes(extension)) {
            resource_type = "audio";
        }

        const options = {
            resource_type,
            // Use the provided subfolder or a generic one
            folder: `drishtiksha/${subfolder || "media"}`,
        };

        const response = await uploadOnCloudinary(localFilePath, options);
        return {
            url: response.secure_url,
            publicId: response.public_id,
        };
    },

    /**
     * Uploads a file from a readable stream to Cloudinary.
     * @param {ReadableStream} stream - The readable stream of the file content.
     * @param {object} options - Options for the upload (e.g., folder, resource_type).
     * @returns {Promise<{url: string, publicId: string}>} - The public URL and ID of the uploaded file.
     */
    async uploadStream(stream, options) {
        // --- UPDATED: The options object (including resource_type) is now passed directly ---
        // to the underlying Cloudinary utility, making it flexible for any stream type.
        const response = await uploadStreamToCloudinary(stream, options);
        return {
            url: response.secure_url,
            publicId: response.public_id,
        };
    },

    /**
     * Deletes a file from Cloudinary.
     * @param {string} publicId - The public ID of the file to delete.
     * @param {string} [resourceType="video"] - The type of the resource ('video', 'image', 'audio').
     */
    async deleteFile(publicId, resourceType = "video") {
        // The resourceType is now dynamically passed from the mediaService,
        // ensuring the correct type of asset is deleted from Cloudinary.
        await deleteFromCloudinary(publicId, resourceType);
    },
};

export default cloudinaryProvider;
