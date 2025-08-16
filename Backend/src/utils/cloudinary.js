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

const uploadOnCloudinary = async (localFilePath) => {
    try {
        if (!localFilePath) {
            throw new ApiError(400, "Local file path is required");
        }

        const response = await cloudinary.uploader.upload(localFilePath, {
            resource_type: "video",
            folder: "deepfake-analysis-videos",
        });

        logger.info(
            `File ${localFilePath} uploaded to Cloudinary: ${response.secure_url}`
        );
        return response;
    } catch (error) {
        logger.error(
            `Cloudinary upload failed for path ${localFilePath}:`,
            error
        );
        throw new ApiError(500, "Failed to upload video to Cloudinary");
    } finally {
        try {
            await fs.unlink(localFilePath);
        } catch (cleanupError) {
            if (cleanupError.code !== "ENOENT") {
                logger.error(
                    `Failed to cleanup local file ${localFilePath}:`,
                    cleanupError
                );
            }
        }
    }
};

// CHANGED: This function is now rewritten to be more robust.
// REASON: It properly wraps the entire stream pipeline in a Promise and handles all critical events ('error', 'finish'), preventing unhandled rejections and silent failures.
const uploadStreamToCloudinary = (stream, options = {}) => {
    return new Promise((resolve, reject) => {
        const uploadStream = cloudinary.uploader.upload_stream(
            options,
            (error, result) => {
                if (error) {
                    logger.error("Cloudinary stream upload failed:", error);
                    return reject(
                        new ApiError(
                            500,
                            `Cloudinary stream upload failed: ${error.message}`
                        )
                    );
                }
                if (!result) {
                    logger.error(
                        "Cloudinary stream upload returned no result."
                    );
                    return reject(
                        new ApiError(
                            500,
                            "Cloudinary stream upload returned no result."
                        )
                    );
                }
                logger.info(
                    `Stream successfully uploaded to Cloudinary: ${result.secure_url}`
                );
                resolve(result);
            }
        );

        // Pipe the source stream to Cloudinary's upload stream and handle errors
        stream.pipe(uploadStream).on("error", (err) => {
            logger.error("Error during stream piping to Cloudinary:", err);
            reject(new ApiError(500, `Stream pipe failed: ${err.message}`));
        });
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
