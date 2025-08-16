import { promises as fs, createWriteStream } from "fs";
import path from "path";
import { videoRepository } from "../repositories/video.repository.js";
import { modelAnalysisService } from "./modelAnalysis.service.js";
import { addVideoToQueue } from "../queue/videoProcessorQueue.js";
import {
    uploadOnCloudinary,
    deleteFromCloudinary,
} from "../utils/cloudinary.js";
import { ApiError } from "../utils/ApiError.js";
import logger from "../utils/logger.js";
import axios from "axios";

const ADVANCED_MODELS = ["SIGLIP_LSTM_V3", "COLOR_CUES_LSTM_V1"];

class VideoService {
    async createVideoAndQueueForAnalysis(file, user, description) {
        if (!file) {
            throw new ApiError(400, "A video file is required.");
        }

        logger.info(
            `Uploading video file ${file.originalname} to Cloudinary...`
        );
        const cloudinaryResponse = await uploadOnCloudinary(file.path);
        if (!cloudinaryResponse || !cloudinaryResponse.public_id) {
            throw new ApiError(500, "Failed to upload video to cloud storage.");
        }

        const videoData = {
            filename: file.originalname,
            description,
            url: cloudinaryResponse.secure_url,
            publicId: cloudinaryResponse.public_id,
            mimetype: file.mimetype,
            size: file.size,
            status: "QUEUED",
            userId: user.id,
        };

        const newVideo = await videoRepository.create(videoData);
        logger.info(`Video record created with ID: ${newVideo.id}`);

        addVideoToQueue(newVideo.id);

        return newVideo;
    }

    async runAllAnalysesForVideo(videoId) {
        await videoRepository.updateStatus(videoId, "PROCESSING");
        logger.info(`Started processing video ID: ${videoId}`);

        const video = await videoRepository.findById(videoId);
        if (!video) throw new Error(`Video ${videoId} not found.`);

        let localVideoPath;
        let healthData = null;

        try {
            const healthStatus = await modelAnalysisService.getHealthStatus();
            healthData = healthStatus;

            // Store server health information for monitoring
            try {
                await videoRepository.storeServerHealth({
                    serverUrl: healthStatus.serverUrl,
                    status: healthStatus.status || "HEALTHY",
                    availableModels:
                        healthStatus.active_models?.map((m) => m.name) || [],
                    modelStates: healthStatus.active_models || null,
                    loadMetrics: healthStatus.load_metrics || null,
                    gpuInfo: healthStatus.gpu_info || null,
                    systemResources: healthStatus.system_resources || null,
                    responseTime: healthStatus.responseTime,
                    uptime: healthStatus.uptime,
                    version: healthStatus.version,
                });
            } catch (healthError) {
                logger.warn(
                    `Failed to store server health: ${healthError.message}`
                );
            }

            const availableModels = healthStatus.active_models
                .filter((m) => m.loaded)
                .map((m) => m.name);

            if (availableModels.length === 0) {
                throw new Error(
                    "No models are currently available on the analysis server."
                );
            }
            logger.info(
                `Found available models: ${availableModels.join(", ")}`
            );

            localVideoPath = await this._downloadVideo(video.url, videoId);

            // Use comprehensive analysis as the default approach
            for (const modelName of availableModels) {
                const modelEnum =
                    modelAnalysisService.mapModelNameToEnum(modelName);
                if (!modelEnum) continue;

                // For advanced models, run comprehensive analysis
                if (ADVANCED_MODELS.includes(modelEnum)) {
                    await this._runAndSaveComprehensiveAnalysis(
                        videoId,
                        localVideoPath,
                        modelName,
                        modelEnum,
                        healthData
                    );
                } else {
                    // For basic models, fall back to quick analysis only
                    await this._runAndSaveAnalysis(
                        videoId,
                        localVideoPath,
                        "QUICK",
                        modelName,
                        modelEnum
                    );
                }
            }

            await videoRepository.updateStatus(videoId, "ANALYZED");
            logger.info(
                `✅ Successfully completed all analyses for video ID: ${videoId}`
            );
        } catch (error) {
            logger.error(
                `A critical error occurred during analysis for video ${videoId}: ${error.message}`
            );
            await this.markVideoAsFailed(videoId, error.message);
        } finally {
            if (localVideoPath) {
                await fs
                    .unlink(localVideoPath)
                    .catch((err) =>
                        logger.error(
                            `Failed to cleanup temp file: ${err.message}`
                        )
                    );
                // logger.info(`Cleaned up temporary file: ${localVideoPath}`);
            }
        }
    }

    async _runAndSaveComprehensiveAnalysis(
        videoId,
        videoPath,
        modelName,
        modelEnum,
        healthData = null
    ) {
        try {
            logger.info(
                `Running comprehensive analysis for video ${videoId} with model ${modelName}`
            );

            // Run comprehensive analysis with frames and visualization included
            const result = await modelAnalysisService.analyzeVideoComprehensive(
                videoPath,
                modelName,
                videoId,
                true, // includeFrames
                true // includeVisualization
            );

            // Enhance result with health data if available
            if (healthData) {
                // Find model-specific information from health data
                const modelInfo = healthData.active_models?.find(
                    (m) => m.name === modelName
                );
                if (modelInfo) {
                    result.modelInfo = {
                        ...result.modelInfo,
                        model_name: modelInfo.name,
                        device: modelInfo.device,
                        loaded: modelInfo.loaded,
                        memory_usage: modelInfo.memory_usage,
                        load_time: modelInfo.load_time,
                    };
                }

                // Add system information from health data
                result.systemInfo = {
                    ...result.systemInfo,
                    gpu_info: healthData.gpu_info,
                    system_resources: healthData.system_resources,
                    load_metrics: healthData.load_metrics,
                    server_version: healthData.version,
                    uptime: healthData.uptime,
                };

                result.serverInfo = {
                    version: healthData.version,
                    status: healthData.status,
                    response_time: healthData.responseTime,
                    active_models_count: healthData.active_models?.length || 0,
                };
            }

            // Save the main comprehensive analysis result
            await videoRepository.createAnalysisResult(videoId, {
                ...result,
                analysisType: "COMPREHENSIVE",
            });

            // If visualization was generated, upload it and save the URL
            if (result.visualizationGenerated && result.visualizationFilename) {
                const visualizationPath = path.join(
                    "uploads",
                    "visualizations",
                    result.visualizationFilename
                );

                // Check if the file exists before uploading
                try {
                    await fs.access(visualizationPath);
                    const cloudinaryResponse = await uploadOnCloudinary(
                        visualizationPath
                    );

                    if (cloudinaryResponse) {
                        // Update the comprehensive analysis record with the visualization URL
                        await videoRepository.createAnalysisResult(videoId, {
                            model: modelEnum,
                            analysisType: "VISUALIZE",
                            visualizedUrl: cloudinaryResponse.secure_url,
                            prediction: "REAL", // Valid enum value for successful visualization
                            confidence: 1.0,
                        });

                        // Clean up local file
                        await fs.unlink(visualizationPath);
                        logger.info(
                            `Uploaded and cleaned up visualization for video ${videoId}`
                        );
                    }
                } catch (visualError) {
                    logger.warn(
                        `Visualization file not found or upload failed for video ${videoId}: ${visualError.message}`
                    );
                }
            }

            logger.info(
                `✅ Comprehensive analysis completed for video ${videoId} with model ${modelName}`
            );
        } catch (error) {
            logger.error(
                `[${modelName}/COMPREHENSIVE] analysis failed for video ${videoId}: ${error.message}`
            );
            await videoRepository.createAnalysisError(
                videoId,
                modelEnum,
                "COMPREHENSIVE",
                error
            );
        }
    }

    async _runAndSaveAnalysis(
        videoId,
        videoPath,
        analysisType,
        modelName,
        modelEnum
    ) {
        try {
            const result = await modelAnalysisService.analyzeVideo(
                videoPath,
                analysisType,
                modelName,
                videoId
            );
            await videoRepository.createAnalysisResult(videoId, {
                ...result,
                analysisType,
            });
        } catch (error) {
            logger.error(
                `[${modelName}/${analysisType}] analysis failed for video ${videoId}: ${error.message}`
            );
            await videoRepository.createAnalysisError(
                videoId,
                modelEnum,
                analysisType,
                error
            );
        }
    }

    async _runAndSaveVisualization(videoId, videoPath, modelName, modelEnum) {
        let visualResult;
        try {
            visualResult = await modelAnalysisService.generateVisualAnalysis(
                videoPath,
                modelName,
                videoId
            );
            const cloudinaryResponse = await uploadOnCloudinary(
                visualResult.visualizationPath
            );
            if (!cloudinaryResponse)
                throw new Error(
                    "Failed to upload visualization to Cloudinary."
                );

            // For visualization, we need a valid prediction enum value
            // Since visualization doesn't produce a prediction itself, use "REAL" as default
            // This represents that the visualization was successfully generated
            await videoRepository.createAnalysisResult(videoId, {
                model: modelEnum,
                analysisType: "VISUALIZE",
                visualizedUrl: cloudinaryResponse.secure_url,
                prediction: "REAL", // Use valid enum value instead of "N/A"
                confidence: 1.0, // Full confidence that visualization was generated
            });

            // Clean up the local file after everything is successful
            try {
                await fs.unlink(visualResult.visualizationPath);
            } catch (unlinkError) {
                // File might already be deleted, just log and continue
                logger.warn(
                    `File ${visualResult.visualizationPath} already deleted or not found: ${unlinkError.message}`
                );
            }
        } catch (error) {
            logger.error(
                `[${modelName}/VISUALIZE] analysis failed for video ${videoId}: ${error.message}`
            );
            await videoRepository.createAnalysisError(
                videoId,
                modelEnum,
                "VISUALIZE",
                error
            );

            // Clean up the local file if it exists
            if (visualResult?.visualizationPath) {
                try {
                    await fs.access(visualResult.visualizationPath);
                    await fs.unlink(visualResult.visualizationPath);
                } catch (cleanupError) {
                    // File doesn't exist or already deleted, that's fine
                    logger.debug(
                        `Cleanup: File ${visualResult.visualizationPath} not found or already deleted`
                    );
                }
            }
        }
    }

    async _downloadVideo(videoUrl, videoId) {
        const tempDir = path.join("temp");
        await fs.mkdir(tempDir, { recursive: true });
        const tempFilePath = path.join(
            tempDir,
            `video-${videoId}-${Date.now()}.mp4`
        );

        logger.info(`Downloading video for analysis to ${tempFilePath}`);
        const writer = createWriteStream(tempFilePath);

        const response = await axios({
            url: videoUrl,
            method: "GET",
            responseType: "stream",
        });

        response.data.pipe(writer);

        return new Promise((resolve, reject) => {
            writer.on("finish", () => resolve(tempFilePath));
            writer.on("error", (err) => {
                logger.error(
                    `Failed to download video ${videoId}: ${err.message}`
                );
                reject(err);
            });
        });
    }

    async markVideoAsFailed(videoId, errorMessage) {
        await videoRepository.updateStatus(videoId, "FAILED");
        logger.error(
            `Marked video ${videoId} as FAILED. Reason: ${errorMessage}`
        );
    }

    async getAllVideosForUser(userId) {
        return videoRepository.findAllByUserId(userId);
    }

    async getVideoWithAnalyses(videoId, userId) {
        const video = await videoRepository.findByIdAndUserId(videoId, userId);
        if (!video) {
            throw new ApiError(
                404,
                "Video not found or you do not have permission to view it."
            );
        }
        return video;
    }

    async deleteVideoById(videoId, userId) {
        const video = await this.getVideoWithAnalyses(videoId, userId);
        if (video.publicId) {
            await deleteFromCloudinary(video.publicId, "video");
            logger.info(`Deleted video ${video.publicId} from Cloudinary.`);
        }
        await videoRepository.deleteById(videoId);
    }
}

export const videoService = new VideoService();
