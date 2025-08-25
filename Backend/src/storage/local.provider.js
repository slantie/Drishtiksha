// src/storage/local.provider.js

import { promises as fs, createWriteStream } from "fs";
import path from "path";
import logger from "../utils/logger.js";
import { ApiError } from "../utils/ApiError.js";

const STORAGE_ROOT = process.env.LOCAL_STORAGE_PATH || "public/media";
const BASE_URL = process.env.BASE_URL || "http://localhost:3000";

const ensureDirectoryExists = async (dirPath) => {
    try {
        await fs.mkdir(dirPath, { recursive: true });
    } catch (error) {
        throw new ApiError(
            500,
            `Could not create storage directory: ${error.message}`
        );
    }
};

const localProvider = {
    /**
     * Moves a file from a temporary path to the permanent local storage.
     * @param {string} localFilePath - The temporary local path of the file.
     * @param {string} [subfolder="media"] - The subfolder to store the file in (e.g., 'videos', 'images').
     * @returns {Promise<{url: string, publicId: string}>} - The public URL and relative path (as publicId).
     */
    // UPDATED: The default subfolder is now more generic.
    async uploadFile(localFilePath, subfolder = "media") {
        const permanentStorageDir = path.join(STORAGE_ROOT, subfolder);
        await ensureDirectoryExists(permanentStorageDir);

        const uniqueFilename = `${Date.now()}-${path.basename(localFilePath)}`;
        const destinationPath = path.join(permanentStorageDir, uniqueFilename);

        try {
            await fs.rename(localFilePath, destinationPath);
        } catch (error) {
            logger.warn(
                `fs.rename failed: ${error.message}. Falling back to copy/unlink.`
            );
            await fs.copyFile(localFilePath, destinationPath);
            await fs.unlink(localFilePath);
        }

        const relativePath = path.join(subfolder, uniqueFilename);
        const urlPath = STORAGE_ROOT.replace(/^public\//, "").replace(
            /\\/g,
            "/"
        );
        const publicUrl = `${BASE_URL}/${urlPath}/${relativePath}`
            .replace(/\/+/g, "/")
            .replace(":/", "://");

        return {
            url: publicUrl,
            publicId: relativePath,
        };
    },

    /**
     * Saves a file from a readable stream to local storage.
     * @param {ReadableStream} stream - The readable stream of the file content.
     * @param {object} options - Options for the upload, including folder and resource_type.
     * @returns {Promise<{url: string, publicId: string}>} - The public URL and relative path.
     */
    async uploadStream(stream, options = {}) {
        const subfolder = options.folder || "visualizations";
        const permanentStorageDir = path.join(STORAGE_ROOT, subfolder);
        await ensureDirectoryExists(permanentStorageDir);

        // --- NEW: Determine file extension based on resource type for streams ---
        const extension = options.resource_type === "image" ? ".png" : ".mp4";
        const uniqueFilename = `${Date.now()}-${typeLabel}${extension}`;
        const destinationPath = path.join(permanentStorageDir, uniqueFilename);

        return new Promise((resolve, reject) => {
            const writeStream = createWriteStream(destinationPath);
            stream.pipe(writeStream);
            writeStream.on("finish", () => {
                const relativePath = path.join(subfolder, uniqueFilename);
                const urlPath = STORAGE_ROOT.replace(/^public\//, "").replace(
                    /\\/g,
                    "/"
                );
                const publicUrl = `${BASE_URL}/${urlPath}/${relativePath}`
                    .replace(/\/+/g, "/")
                    .replace(":/", "://");
                resolve({ url: publicUrl, publicId: relativePath });
            });
            writeStream.on("error", (error) => {
                reject(
                    new ApiError(
                        500,
                        `Failed to save stream to local storage: ${error.message}`
                    )
                );
            });
        });
    },

    /**
     * Deletes a file from the local filesystem.
     * @param {string} publicId - The relative path of the file to delete.
     * @param {string} [resourceType] - The type of resource (ignored by local provider, but included for interface consistency).
     */
    async deleteFile(publicId, resourceType) {
        if (!publicId) return;
        const fullPath = path.join(STORAGE_ROOT, publicId);
        try {
            await fs.unlink(fullPath);
            logger.info(`Successfully deleted local file: ${fullPath}`);
        } catch (error) {
            if (error.code === "ENOENT") {
                logger.warn(
                    `Attempted to delete a non-existent file: ${fullPath}`
                );
            } else {
                logger.error(
                    `Failed to delete local file ${fullPath}: ${error.message}`
                );
            }
        }
    },
};

export default localProvider;
