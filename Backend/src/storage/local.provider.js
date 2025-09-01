// src/storage/local.provider.js

import { promises as fs, createWriteStream, existsSync } from 'fs';
import path from 'path';
import { config } from '../config/env.js';
import logger from '../utils/logger.js';
import { ApiError } from '../utils/ApiError.js';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const STORAGE_ROOT = config.LOCAL_STORAGE_PATH;
const BASE_URL = config.BASE_URL;
const PROJECT_ROOT = path.resolve(__dirname, '..', '..');

const ensureDirectoryExists = async (dirPath) => {
    try {
        await fs.mkdir(dirPath, { recursive: true });
    } catch (error) {
        throw new ApiError(500, `Could not create storage directory: ${error.message}`);
    }
};

const localProvider = {
    async uploadFile(localFilePath, subfolder = 'media') {
        const permanentStorageDir = path.join(PROJECT_ROOT, STORAGE_ROOT, subfolder);
        await ensureDirectoryExists(permanentStorageDir);

        const uniqueFilename = `${Date.now()}-${path.basename(localFilePath)}`;
        const destinationPath = path.join(permanentStorageDir, uniqueFilename);

        try {
            await fs.rename(localFilePath, destinationPath);
        } catch (error) {
            logger.warn(`fs.rename failed for ${localFilePath}: ${error.message}. Falling back to copy/unlink.`);
            await fs.copyFile(localFilePath, destinationPath);
            await fs.unlink(localFilePath);
        }

        const relativePath = path.join(subfolder, uniqueFilename).replace(/\\/g, '/');
        const urlPath = STORAGE_ROOT.replace(/^public\//, '').replace(/\\/g, '/');
        const publicUrl = new URL(`${urlPath}/${relativePath}`, BASE_URL).href;
        
        return { url: publicUrl, publicId: relativePath };
    },

    async uploadStream(stream, options = {}) {
        const subfolder = options.folder || 'visualizations';
        const permanentStorageDir = path.join(PROJECT_ROOT, STORAGE_ROOT, subfolder);
        await ensureDirectoryExists(permanentStorageDir);

        const extension = options.resource_type === 'image' ? '.png' : '.mp4';
        const uniqueFilename = `${Date.now()}-visualization${extension}`;
        const destinationPath = path.join(permanentStorageDir, uniqueFilename);

        return new Promise((resolve, reject) => {
            const writeStream = createWriteStream(destinationPath);
            stream.pipe(writeStream);
            writeStream.on('finish', () => {
                const relativePath = path.join(subfolder, uniqueFilename).replace(/\\/g, '/');
                const urlPath = STORAGE_ROOT.replace(/^public\//, '').replace(/\\/g, '/');
                const publicUrl = new URL(`${urlPath}/${relativePath}`, BASE_URL).href;
                resolve({ url: publicUrl, publicId: relativePath });
            });
            writeStream.on('error', (error) => {
                reject(new ApiError(500, `Failed to save stream to local storage: ${error.message}`));
            });
        });
    },
    
    async deleteFile(publicId) {
        if (!publicId) return;
        const fullPath = path.join(PROJECT_ROOT, STORAGE_ROOT, publicId);
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