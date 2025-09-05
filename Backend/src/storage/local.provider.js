// Backend/src/storage/local.provider.js

import { promises as fs, existsSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { config } from "../config/env.js";
import logger from "../utils/logger.js";
import { ApiError } from "../utils/ApiError.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PROJECT_ROOT = path.resolve(__dirname, "..", "..");
const STORAGE_ROOT = path.join(PROJECT_ROOT, config.LOCAL_STORAGE_PATH);

const ensureDirectoryExists = async (dirPath) => {
  try {
    if (!existsSync(dirPath)) {
      await fs.mkdir(dirPath, { recursive: true });
    }
  } catch (error) {
    throw new ApiError(
      500,
      `Could not create storage directory: ${error.message}`
    );
  }
};

const localProvider = {
  async uploadFile(localFilePath, subfolder = "uploads") {
    const permanentStorageDir = path.join(STORAGE_ROOT, subfolder);
    await ensureDirectoryExists(permanentStorageDir);

    const uniqueFilename = `${Date.now()}-${path.basename(localFilePath)}`;
    const destinationPath = path.join(permanentStorageDir, uniqueFilename);

    try {
      await fs.rename(localFilePath, destinationPath);
    } catch (error) {
      if (error.code === "EXDEV") {
        logger.warn(
          `fs.rename failed (cross-device), falling back to copy/unlink.`
        );
        await fs.copyFile(localFilePath, destinationPath);
        await fs.unlink(localFilePath);
      } else {
        throw error;
      }
    }

    const publicId = path.join(subfolder, uniqueFilename).replace(/\\/g, "/");

    // Construct the full, public URL using the ASSETS_BASE_URL from config
    // e.g., 'http://localhost:3001' + '/media/' + 'videos/123.mp4'
    const publicUrl = new URL(
      `${config.LOCAL_STORAGE_PATH.split("/").slice(1).join("/")}/${publicId}`,
      config.ASSETS_BASE_URL
    ).href;

    return { url: publicUrl, publicId: publicId };
  },

  async deleteFile(publicId) {
    if (!publicId) return;
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
