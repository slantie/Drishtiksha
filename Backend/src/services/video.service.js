// src/services/video.service.js

// import { videoRepository } from "../repositories/video.repository.js";
// import {
//     uploadOnCloudinary,
//     deleteFromCloudinary,
// } from "../utils/cloudinary.js";
// import { ApiError } from "../utils/ApiError.js";
// import { videoProcessorQueue } from "../queue/videoProcessorQueue.js";
// import { modelAnalysisService } from "./modelAnalysis.service.js";
// import logger from "../utils/logger.js";
// import axios from "axios";
// import { fileURLToPath } from "url";
// import fs from "fs/promises";
// import path from "path";

import { promises as fs } from "fs"; // Import the promise-based API as 'fs'
import { createWriteStream } from "fs"; // Import createWriteStream separately
import path from "path";
import { fileURLToPath } from "url";
import { videoRepository } from "../repositories/video.repository.js";
import { uploadOnCloudinary, uploadStreamToCloudinary, deleteFromCloudinary } from "../utils/cloudinary.js";
import { ApiError } from "../utils/ApiError.js";
import { videoProcessorQueue } from "../queue/videoProcessorQueue.js";
import logger from "../utils/logger.js";
import { modelAnalysisService } from "./modelAnalysis.service.js";
import axios from "axios";


const ANALYSIS_MODELS = ["LSTM_SIGLIP", "RPPG", "COLORCUES"];

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.join(__dirname, '..', '..');

const generateMockAnalysis = (filename, fileSize, model) => {
    const isLikelyReal = Math.random() > 0.3;
    const baseConfidence = isLikelyReal
        ? 0.75 + Math.random() * 0.2
        : 0.6 + Math.random() * 0.3;
    const confidence = parseFloat(
        Math.min(0.98, Math.max(0.52, baseConfidence)).toFixed(4)
    );
    const prediction = confidence > 0.75 ? "REAL" : "FAKE";
    const processingTime = parseFloat(
        (
            2.5 +
            (fileSize / (1024 * 1024)) * 0.1 +
            (Math.random() - 0.5)
        ).toFixed(2)
    );
    return {
        prediction,
        confidence,
        processingTime,
        model,
        status: "COMPLETED",
    };
};

export const videoService = {
    async uploadAndProcessVideo(file, description, user) {
        if (!file) {
            throw new ApiError(400, "No video file provided");
        }

        const cloudinaryResponse = await uploadOnCloudinary(file.path);
        if (!cloudinaryResponse) {
            throw new ApiError(500, "Video failed to upload on Cloudinary");
        }

        const newVideo = await videoRepository.create({
            filename: file.originalname,
            mimetype: file.mimetype,
            size: file.size,
            description: description || "",
            url: cloudinaryResponse.secure_url,
            publicId: cloudinaryResponse.public_id,
            userId: user.id,
            status: "UPLOADED",
        });

        // Add job to the queue for automatic multi-model analysis
        videoProcessorQueue({ videoId: newVideo.id, userId: newVideo.userId });
        logger.info(
            `Video ${newVideo.id} added to in-memory processing queue.`
        );

        return newVideo;
    },

    async getAllVideosForUser(user) {
        if (user.role === "ADMIN") {
            return videoRepository.findAll();
        }
        return videoRepository.findAllByUser(user.id);
    },

    async getVideoById(videoId, user) {
        const video = await videoRepository.findById(videoId);
        if (!video) {
            throw new ApiError(404, "Video not found");
        }
        if (video.userId !== user.id && user.role !== "ADMIN") {
            throw new ApiError(
                403,
                "Access denied. You do not own this video."
            );
        }
        return video;
    },

    async updateVideoDetails(videoId, updateData, user) {
        const video = await this.getVideoById(videoId, user); // Reuse permission check
        return videoRepository.update(videoId, updateData);
    },

    async deleteVideoById(videoId, user) {
        const video = await this.getVideoById(videoId, user); // Reuse permission check

        if (video.publicId) {
            await deleteFromCloudinary(video.publicId, "video");
        }

        await videoRepository.delete(videoId);
    },

    async runFullAnalysis(videoId) {
        logger.info(`Starting multi-model analysis for video ID: ${videoId}`);
        const video = await videoRepository.findById(videoId);
        if (!video) {
            logger.error(`Video with ID ${videoId} not found for analysis.`);
            return;
        }

        await videoRepository.update(videoId, { status: "PROCESSING" });

        try {
            for (const model of ANALYSIS_MODELS) {
                const existing = video.analyses.find((a) => a.model === model);
                if (existing) {
                    logger.info(
                        `Analysis for model ${model} already exists for video ${videoId}. Skipping.`
                    );
                    continue;
                }

                let results;

                if (model === "LSTM_SIGLIP") {
                    // Use real LSTM model analysis
                    try {
                        logger.info(
                            `Running LSTM model analysis for video ${videoId}.`
                        );

                        // Get the local video file path for analysis
                        const videoPath = await this.getLocalVideoPath(video);

                        results =
                            await modelAnalysisService.analyzeVideoWithFallback(
                                videoPath,
                                videoId,
                                { filename: video.filename, size: video.size }
                            );

                        // Clean up local file if it was downloaded
                        await this.cleanupLocalVideoPath(videoPath, video);
                    } catch (error) {
                        logger.error(
                            `LSTM analysis failed for video ${videoId}: ${error.message}`
                        );
                        // Fall back to mock data for LSTM model
                        results = generateMockAnalysis(
                            video.filename,
                            video.size,
                            model
                        );
                    }
                } else {
                    // Use mock analysis for other models
                    logger.info(
                        `Running mock analysis for video ${videoId} with model ${model}.`
                    );
                    await new Promise((resolve) =>
                        setTimeout(resolve, 1000 + Math.random() * 1500)
                    );
                    results = generateMockAnalysis(
                        video.filename,
                        video.size,
                        model
                    );
                }

                await videoRepository.createAnalysis({ videoId, ...results });
            }
            await videoRepository.update(videoId, { status: "ANALYZED" });
            logger.info(`All analyses for video ${videoId} are now complete.`);
        } catch (error) {
            logger.error(
                `Failed to complete video analysis for ID ${videoId}: ${error.message}`
            );
            await videoRepository.update(videoId, { status: "FAILED" });
        }
    },

    /**
     * Gets local video path for analysis - downloads from Cloudinary if needed
     * @param {Object} video - Video object with URL and metadata
     * @returns {string} Local file path
     */
    async getLocalVideoPath(video) {
        const fs = await import("fs");
        const https = await import("https");
        const http = await import("http");
        const { promisify } = await import("util");
        const pipeline = promisify((await import("stream")).pipeline);

        // Check for local file first (development setup)
        const uploadsDir = path.join(process.cwd(), "uploads", "videos");
        const possibleLocalPaths = [
            path.join(uploadsDir, `video-${video.id}.mp4`),
            path.join(uploadsDir, video.filename),
            path.join(uploadsDir, `${video.id}.mp4`),
        ];

        for (const localPath of possibleLocalPaths) {
            if (fs.existsSync(localPath)) {
                logger.info(`Using local video file: ${localPath}`);
                return localPath;
            }
        }

        // If no local file found, download from Cloudinary
        logger.info(
            `Local video file not found for ${video.id}, downloading from Cloudinary`
        );

        if (!video.url) {
            throw new Error(`No URL available for video ${video.id}`);
        }

        // Create temporary file path
        const tempDir = path.join(process.cwd(), "temp");
        if (!fs.existsSync(tempDir)) {
            fs.mkdirSync(tempDir, { recursive: true });
        }

        const tempFilePath = path.join(
            tempDir,
            `temp-${video.id}-${Date.now()}.mp4`
        );

        try {
            // Download video from Cloudinary
            const httpModule = video.url.startsWith("https:") ? https : http;

            await new Promise((resolve, reject) => {
                const request = httpModule.get(video.url, (response) => {
                    if (response.statusCode !== 200) {
                        reject(
                            new Error(
                                `Failed to download video: HTTP ${response.statusCode}`
                            )
                        );
                        return;
                    }

                    const writeStream = fs.createWriteStream(tempFilePath);
                    pipeline(response, writeStream).then(resolve).catch(reject);
                });

                request.on("error", reject);
                request.setTimeout(30000, () => {
                    request.destroy();
                    reject(new Error("Download timeout"));
                });
            });

            logger.info(`Video downloaded successfully to: ${tempFilePath}`);
            return tempFilePath;
        } catch (error) {
            // Clean up partial download
            if (fs.existsSync(tempFilePath)) {
                fs.unlinkSync(tempFilePath);
            }
            logger.error(
                `Failed to download video ${video.id}: ${error.message}`
            );
            throw error;
        }
    },

    /**
     * Cleans up temporary video files if they were downloaded
     * @param {string} videoPath - Path to the video file
     * @param {Object} video - Video object
     */
    async cleanupLocalVideoPath(videoPath, video) {
        const fs = await import("fs");

        // Only clean up temporary files (those in temp directory)
        if (videoPath.includes("temp") && fs.existsSync(videoPath)) {
            try {
                fs.unlinkSync(videoPath);
                logger.debug(`Cleaned up temporary video file: ${videoPath}`);
            } catch (error) {
                logger.warn(
                    `Failed to cleanup temporary file ${videoPath}: ${error.message}`
                );
            }
        }
    },
    
    async createVisualAnalysis(videoId, user) {
        const video = await this.getVideoById(videoId, user); // Reuse permission check

        if (!video.url) {
            throw new ApiError(400, "Original video URL not found. Cannot perform visual analysis.");
        }

        const tempDir = path.join(projectRoot, 'temp');
        await fs.mkdir(tempDir, { recursive: true });

        const originalVideoTempPath = path.join(tempDir, `original_${videoId}.mp4`);
        
        try {
            // 1. Download the original video from Cloudinary
            logger.info(`Downloading original video from Cloudinary: ${video.url}`);
            const writer = createWriteStream(originalVideoTempPath);// Note: createWriteStream is from the core 'fs'
            const response = await axios({ url: video.url, method: 'GET', responseType: 'stream' });
            response.data.pipe(writer);
            await new Promise((resolve, reject) => {
                writer.on('finish', resolve);
                writer.on('error', reject);
            });
            logger.info(`Successfully downloaded original video to: ${originalVideoTempPath}`);

            // 2. Send the local file to the Python service
            const visualAnalysisStream = await modelAnalysisService.generateVisualAnalysis(originalVideoTempPath);

            // 3. Upload the resulting stream directly to Cloudinary
            let uploadResult;
            try {
                logger.info(`Uploading visualized analysis stream to Cloudinary for video: ${videoId}`);
                uploadResult = await uploadStreamToCloudinary(visualAnalysisStream, {
                    resource_type: "video",
                    folder: "visual_analyses",
                    public_id: `visual_${video.publicId}`
                });
            } catch (uploadError) {
                // This will catch the rejection from the promise in uploadStreamToCloudinary
                logger.error(`Caught an error during Cloudinary stream upload for video ${videoId}:`, uploadError);
                throw new ApiError(500, "The generated visual analysis failed to upload.");
            }
            if (!uploadResult || !uploadResult.secure_url) {
                throw new ApiError(500, "Upload of visualized video succeeded but returned no URL.");
            }
            
            return await videoRepository.update(videoId, {
                visualizedUrl: uploadResult.secure_url,
            });

        } finally {
            // 5. Clean up the temporary local file using an async try/catch block
            try {
                await fs.unlink(originalVideoTempPath);
                logger.info(`Cleaned up temporary file: ${originalVideoTempPath}`);
            } catch (cleanupError) {
                // If the file doesn't exist, the error code will be 'ENOENT'.
                // We can safely ignore this error, as it means the file is already gone.
                // For any other error, we should log it.
                if (cleanupError.code !== 'ENOENT') {
                    logger.error(`Error cleaning up temporary file ${originalVideoTempPath}:`, cleanupError);
                }
            }
        }
    }
};
