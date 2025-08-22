// src/storage/cloudinary.provider.js

import {
    uploadOnCloudinary,
    uploadStreamToCloudinary,
    deleteFromCloudinary,
} from "../utils/cloudinary.js";

/**
 * Cloudinary storage provider.
 * This object standardizes the interface for interacting with Cloudinary.
 */
const cloudinaryProvider = {
    /**
     * Uploads a file from a local path to Cloudinary.
     * @param {string} localFilePath - The temporary local path of the file.
     * @returns {Promise<{url: string, publicId: string}>} - The public URL and ID of the uploaded file.
     */
    async uploadFile(localFilePath) {
        const response = await uploadOnCloudinary(localFilePath);
        return {
            url: response.secure_url,
            publicId: response.public_id,
        };
    },

    /**
     * Uploads a file from a readable stream to Cloudinary.
     * @param {ReadableStream} stream - The readable stream of the file content.
     * @param {object} options - Options for the upload (e.g., folder).
     * @returns {Promise<{url: string, publicId: string}>} - The public URL and ID of the uploaded file.
     */
    async uploadStream(stream, options) {
        const response = await uploadStreamToCloudinary(stream, options);
        return {
            url: response.secure_url,
            publicId: response.public_id,
        };
    },

    /**
     * Deletes a file from Cloudinary.
     * @param {string} publicId - The public ID of the file to delete.
     * @param {string} [resourceType="video"] - The type of the resource.
     */
    async deleteFile(publicId, resourceType = "video") {
        await deleteFromCloudinary(publicId, resourceType);
    },
};

export default cloudinaryProvider;