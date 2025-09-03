// src/storage/local.provider.js

import { promises as fs, createWriteStream, existsSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { config } from "../config/env.js";
import logger from "../utils/logger.js";
import { ApiError } from "../utils/ApiError.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// This is the absolute path to the project root (e.g., /app in Docker)
const PROJECT_ROOT = path.resolve(__dirname, "..", "..");
// This is the absolute path to the storage directory (e.g., /app/public/media)
const STORAGE_ROOT = path.join(PROJECT_ROOT, config.LOCAL_STORAGE_PATH);

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
  async uploadFile(localFilePath, subfolder = "uploads") {
    // Correctly join the absolute storage root with the desired subfolder.
    const permanentStorageDir = path.join(STORAGE_ROOT, subfolder);
    await ensureDirectoryExists(permanentStorageDir);

    const uniqueFilename = `${Date.now()}-${path.basename(localFilePath)}`;
    const destinationPath = path.join(permanentStorageDir, uniqueFilename);

    // Use a robust copy-then-unlink strategy as a fallback for fs.rename across devices.
    try {
      await fs.rename(localFilePath, destinationPath);
    } catch (error) {
      logger.warn(
        `fs.rename failed (possibly cross-device), falling back to copy/unlink: ${error.message}`
      );
      await fs.copyFile(localFilePath, destinationPath);
      await fs.unlink(localFilePath);
    }

    // This is the relative path from the storage root, used for deletion and database storage.
    const publicId = path.join(subfolder, uniqueFilename).replace(/\\/g, "/");

    // This is the publicly accessible URL served by the dedicated asset server.
    const publicUrl = new URL(publicId, config.ASSETS_BASE_URL).href;

    return { url: publicUrl, publicId: publicId };
  },

  async uploadStream(stream, options = {}) {
    const subfolder = options.folder || "visualizations";
    const permanentStorageDir = path.join(STORAGE_ROOT, subfolder);
    await ensureDirectoryExists(permanentStorageDir);

    const extension = options.resource_type === "image" ? ".png" : ".mp4";
    const uniqueFilename = `${Date.now()}-visualization${extension}`;
    const destinationPath = path.join(permanentStorageDir, uniqueFilename);

    return new Promise((resolve, reject) => {
      const writeStream = createWriteStream(destinationPath);
      stream.pipe(writeStream);
      writeStream.on("finish", () => {
        const publicId = path
          .join(subfolder, uniqueFilename)
          .replace(/\\/g, "/");
        const publicUrl = new URL(publicId, config.ASSETS_BASE_URL).href;
        resolve({ url: publicUrl, publicId: publicId });
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

  async deleteFile(publicId) {
    if (!publicId) return;
    // Construct the absolute path to the file within the storage directory for deletion.
    const fullPath = path.join(STORAGE_ROOT, publicId);
    try {
      if (existsSync(fullPath)) {
        await fs.unlink(fullPath);
        logger.info(`Successfully deleted local file: ${fullPath}`);
      }
    } catch (error) {
      logger.error(`Failed to delete local file ${fullPath}: ${error.message}`);
    }
  },
};

export default localProvider;
