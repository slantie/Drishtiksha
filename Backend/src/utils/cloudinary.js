// src/utils/cloudinary.js

import { v2 as cloudinary } from 'cloudinary';
import { promises as fs } from 'fs';
import { ApiError } from './ApiError.js';
import logger from './logger.js';
import { config } from '../config/env.js';

cloudinary.config({
    cloud_name: config.CLOUDINARY_CLOUD_NAME,
    api_key: config.CLOUDINARY_API_KEY,
    api_secret: config.CLOUDINARY_API_SECRET,
});

export const uploadOnCloudinary = async (localFilePath, options = {}) => {
    try {
        if (!localFilePath) {
            throw new ApiError(400, 'Local file path is required for Cloudinary upload.');
        }
        const response = await cloudinary.uploader.upload(localFilePath, { ...options });
        logger.info(`File ${localFilePath} uploaded to Cloudinary: ${response.secure_url}`);
        return response;
    } catch (error) {
        logger.error(`Cloudinary upload failed for path ${localFilePath}:`, error);
        throw new ApiError(500, `Failed to upload file to Cloudinary: ${error.message}`);
    } finally {
        await fs.unlink(localFilePath).catch(err => {
            if (err.code !== 'ENOENT') {
                logger.error(`Failed to cleanup temp upload file ${localFilePath}:`, err);
            }
        });
    }
};

export const uploadStreamToCloudinary = (stream, options = {}) => {
    return new Promise((resolve, reject) => {
        const uploadStream = cloudinary.uploader.upload_stream(options, (error, result) => {
            if (error) {
                logger.error('Cloudinary stream upload failed:', error);
                return reject(new ApiError(500, `Cloudinary stream upload failed: ${error.message}`));
            }
            if (!result) {
                return reject(new ApiError(500, 'Cloudinary stream upload returned an empty result.'));
            }
            logger.info(`Stream successfully uploaded to Cloudinary: ${result.secure_url}`);
            resolve(result);
        });
        stream.pipe(uploadStream).on('error', (err) => {
            reject(new ApiError(500, `Stream pipe to Cloudinary failed: ${err.message}`));
        });
    });
};

export const deleteFromCloudinary = async (publicId, resourceType = 'video') => {
    try {
        if (!publicId) return null;
        const result = await cloudinary.uploader.destroy(publicId, { resource_type: resourceType });
        logger.info(`Asset ${publicId} deleted from Cloudinary.`);
        return result;
    } catch (error) {
        logger.error(`Cloudinary deletion failed for ${publicId}:`, error);
        throw new ApiError(500, `Failed to delete resource from Cloudinary: ${error.message}`);
    }
};