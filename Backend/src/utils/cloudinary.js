// src/utils/cloudinary.js

import { v2 as cloudinary } from "cloudinary";
import fs from "fs/promises";
import { ApiError } from "./ApiError.js";

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

        // File has been uploaded successfully, now remove the local file
        await fs.unlink(localFilePath);
        return response;
    } catch (error) {
        // If upload fails, still try to remove the local file
        try {
            await fs.unlink(localFilePath);
        } catch (cleanupError) {
            console.error(
                `Failed to cleanup local file ${localFilePath}:`,
                cleanupError
            );
        }
        console.error("Cloudinary upload failed:", error);
        throw new ApiError(500, "Failed to upload video to Cloudinary");
    }
};

const deleteFromCloudinary = async (publicId, resourceType = "video") => {
    try {
        if (!publicId) return null;
        const result = await cloudinary.uploader.destroy(publicId, {
            resource_type: resourceType,
        });
        return result;
    } catch (error) {
        console.error("Cloudinary deletion failed:", error);
        throw new ApiError(500, "Failed to delete resource from Cloudinary");
    }
};

export { uploadOnCloudinary, deleteFromCloudinary };
