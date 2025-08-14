// src/utils/cloudinary.js

import { v2 as cloudinary } from "cloudinary";
import { promises as fs } from "fs";
import { ApiError } from "./ApiError.js";
import logger from "./logger.js";

cloudinary.config({
    cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
    api_key: process.env.CLOUDINARY_API_KEY,
    api_secret: process.env.CLOUDINARY_API_SECRET,
});

/**
 * Uploads a local file to Cloudinary and then deletes the local copy.
 * @param {string} localFilePath - The path to the local file to upload.
 * @returns {Promise<object>} The Cloudinary upload result.
 */
const uploadOnCloudinary = async (localFilePath) => {
    try {
        if (!localFilePath) {
            throw new ApiError(400, "Local file path is required");
        }

        const response = await cloudinary.uploader.upload(localFilePath, {
            resource_type: "video",
            folder: "deepfake-analysis-videos",
        });

        logger.info(`File ${localFilePath} uploaded to Cloudinary: ${response.secure_url}`);
        return response;
    } catch (error) {
        logger.error(`Cloudinary upload failed for path ${localFilePath}:`, error);
        throw new ApiError(500, "Failed to upload video to Cloudinary");
    } finally {
        // This 'finally' block ensures the cleanup happens whether the upload succeeds or fails.
        try {
            await fs.unlink(localFilePath);
            logger.info(`Cleaned up local file: ${localFilePath}`);
        } catch (cleanupError) {
            // Ignore ENOENT error (file already gone), log others.
            if (cleanupError.code !== 'ENOENT') {
                logger.error(`Failed to cleanup local file ${localFilePath}:`, cleanupError);
            }
        }
    }
};

/**
 * Uploads a readable stream to Cloudinary.
 * Used for uploading the generated visual analysis video without saving it locally.
 * @param {object} stream - The readable stream to upload.
 * @param {object} options - Cloudinary upload options (e.g., folder, public_id).
 * @returns {Promise<object>} The Cloudinary upload result.
 */
const uploadStreamToCloudinary = (stream, options = {}) => {
    return new Promise((resolve, reject) => {
        const upload_stream = cloudinary.uploader.upload_stream(
            options,
            (error, result) => {
                if (error) {
                    logger.error("Cloudinary stream upload failed:", error);
                    // Reject the promise so the error can be caught by the calling service
                    return reject(new ApiError(500, "Cloudinary stream upload failed"));
                }
                if (!result) {
                    logger.error("Cloudinary stream upload failed: No result was returned.");
                    return reject(new ApiError(500, "Cloudinary stream upload returned no result."));
                }
                // THIS IS THE LOG WE ARE MISSING. IF THIS APPEARS, THE UPLOAD WAS SUCCESSFUL.
                logger.info(`Stream successfully uploaded to Cloudinary: ${result.secure_url}`);
                resolve(result);
            }
        );

        // Add error handling on the source stream itself
        stream.on('error', (err) => {
            logger.error('Source stream error during Cloudinary upload:', err);
            reject(new ApiError(500, 'The source stream for upload was corrupted or failed.'));
        });

        stream.pipe(upload_stream);
    });
};

const deleteFromCloudinary = async (publicId, resourceType = "video") => {
    try {
        if (!publicId) return null;
        const result = await cloudinary.uploader.destroy(publicId, {
            resource_type: resourceType,
        });
        logger.info(`Asset ${publicId} deleted from Cloudinary.`);
        return result;
    } catch (error) {
        logger.error(`Cloudinary deletion failed for ${publicId}:`, error);
        throw new ApiError(500, "Failed to delete resource from Cloudinary");
    }
};

export { uploadOnCloudinary, uploadStreamToCloudinary, deleteFromCloudinary };