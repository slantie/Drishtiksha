// src/workers/media.worker.js

import { Worker } from "bullmq";
import fs from "fs";
import { promises as fsPromises } from "fs";
import path from "path";
import axios from "axios";
import dotenv from "dotenv";
// RENAMED: Using the new mediaRepository
import { mediaRepository } from "../repositories/media.repository.js";
import { modelAnalysisService } from "../services/modelAnalysis.service.js";
import { redisConnection } from "../config/queue.js";
import logger from "../utils/logger.js";
import { eventService } from "../services/event.service.js";
import storageManager from "../storage/storage.manager.js";
import { toCamelCase } from "../utils/formatKeys.js";

dotenv.config({ path: "./.env" });

const worker = new Worker(
    process.env.MEDIA_PROCESSING_QUEUE_NAME || "media-processing-queue",
    async (job) => {
        logger.info(`[Worker] Picked up job '${job.name}' (ID: ${job.id})`);
        switch (job.name) {
            case "run-single-analysis":
                return await handleSingleAnalysis(job);
            case "finalize-analysis":
                return await handleFinalizeAnalysis(job);
            default:
                throw new Error(`Unknown job name: ${job.name}`);
        }
    },
    { connection: redisConnection, concurrency: 5 }
);

async function handleSingleAnalysis(job) {
    // UPDATED: Destructuring new job data properties
    const { mediaId, mediaType, modelName, serverStats } = job.data;
    let localMediaPath;
    let userId;
    let isTempFile = false;

    try {
        const media = await mediaRepository.findById(mediaId);
        if (!media) {
            logger.warn(
                `[Worker] Media ${mediaId} not found - skipping job (likely cleaned up)`
            );
            return {
                status: "skipped",
                reason: "Media not found - likely cleaned up",
            };
        }
        userId = media.userId;

        await mediaRepository.updateStatus(mediaId, "PROCESSING");
        await eventService.emitProgress({
            mediaId, // Socket event name remains for frontend compatibility
            userId,
            event: "ANALYSIS_STARTED",
            message: `Analysis started for model: ${modelName}`,
            data: { modelName },
        });

        if (process.env.STORAGE_PROVIDER === "local") {
            const localStoragePath =
                process.env.LOCAL_STORAGE_PATH || "public/media";
            localMediaPath = path.resolve(localStoragePath, media.publicId);
            logger.info(
                `[Worker] Using local file for analysis: ${localMediaPath}`
            );
            if (!fs.existsSync(localMediaPath)) {
                throw new Error(`Local file not found at: ${localMediaPath}`);
            }
        } else {
            // UPDATED: Using a generic download function
            localMediaPath = await downloadMedia(
                media.url,
                media.filename,
                mediaId,
                modelName
            );
            isTempFile = true;
            logger.info(
                `[Worker] Downloaded Cloudinary media to temp path: ${localMediaPath}`
            );
        }

        // UPDATED: Passing mediaId and mediaType to the analysis runner
        await runAndSaveComprehensiveAnalysis(
            mediaId,
            mediaType,
            userId,
            localMediaPath,
            modelName,
            serverStats
        );

        await eventService.emitProgress({
            mediaId,
            userId,
            event: "ANALYSIS_COMPLETED",
            message: `Analysis completed for model: ${modelName}`,
            data: { modelName, success: true },
        });
    } catch (error) {
        logger.error(
            `[Worker/Analysis] Job for model ${modelName} on media ${mediaId} failed: ${error.message}`
        );

        // Only try to create analysis error if the media still exists
        try {
            const media = await mediaRepository.findById(mediaId);
            if (media) {
                await mediaRepository.createAnalysisError(
                    mediaId,
                    modelName,
                    "COMPREHENSIVE",
                    error
                );
            }
        } catch (dbError) {
            logger.error(
                `[Worker/Analysis] Failed to save error for ${mediaId}: ${dbError.message}`
            );
        }

        if (userId) {
            await eventService.emitProgress({
                mediaId,
                userId,
                event: "ANALYSIS_COMPLETED",
                message: `Analysis failed for model: ${modelName}.`,
                data: { modelName, success: false, error: error.message },
            });
        }
        throw error;
    } finally {
        if (localMediaPath && isTempFile) {
            await fsPromises
                .unlink(localMediaPath)
                .catch((err) =>
                    logger.error(
                        `[Worker] Cleanup failed for temp file ${localMediaPath}: ${err.message}`
                    )
                );
        }
    }
}

async function handleFinalizeAnalysis(job) {
    // UPDATED: Destructuring mediaId
    const { mediaId } = job.data;
    const media = await mediaRepository.findById(mediaId);
    if (!media) throw new Error(`Cannot finalize, media ${mediaId} not found.`);

    const completedAnalyses = media.analyses.filter(
        (a) => a.status === "COMPLETED"
    ).length;

    // The total number of attempts is now passed in the job data from the service layer
    const totalAnalysesAttempted =
        job.data.totalAnalysesAttempted || media.analyses.length;

    let finalStatus = "FAILED";
    if (
        completedAnalyses === totalAnalysesAttempted &&
        totalAnalysesAttempted > 0
    ) {
        finalStatus = "ANALYZED";
    } else if (completedAnalyses > 0) {
        finalStatus = "PARTIALLY_ANALYZED";
    }

    await mediaRepository.updateStatus(mediaId, finalStatus);
    logger.info(
        `[Finalizer] Finalized media ${mediaId} with status: ${finalStatus} [${completedAnalyses}/${totalAnalysesAttempted} successful]`
    );
}

// REFACTORED: This function now handles different media types and their unique result structures.
async function runAndSaveComprehensiveAnalysis(
    mediaId,
    mediaType,
    userId,
    mediaPath,
    modelName,
    serverStats
) {
    // FIXED: Pass mediaType parameter to the analysis service
    const response = await modelAnalysisService.analyzeMediaComprehensive(
        mediaPath,
        mediaType,
        modelName,
        mediaId,
        userId
    );

    // Recursively convert all keys in the server response to camelCase
    function deepCamelCase(obj) {
        if (Array.isArray(obj)) {
            return obj.map(deepCamelCase);
        } else if (obj && typeof obj === "object") {
            return Object.fromEntries(
                Object.entries(obj).map(([k, v]) => [
                    toCamelCase(k),
                    deepCamelCase(v),
                ])
            );
        }
        return obj;
    }
    const analysisData = deepCamelCase(response.data);
    const modelUsed = response.model_used;

    const { modelInfo, systemInfo } =
        modelAnalysisService.mapServerStatsToDbSchema(serverStats, modelName);

    // --- NEW: Conditionally build the result object based on media type ---
    // This flexible structure directly maps to what the mediaRepository expects.
    const resultToSave = {
        // Common fields for all types
        prediction: analysisData.prediction,
        confidence: analysisData.confidence,
        processingTime: analysisData.processingTime,
        model: modelName,
        modelVersion: modelUsed,
        analysisType: "COMPREHENSIVE",
        modelInfo,
        systemInfo,
    };

    if (mediaType === "VIDEO") {
        resultToSave.metrics = analysisData.metrics;
        resultToSave.framePredictions =
            analysisData.framesAnalysis?.framePredictions;
        resultToSave.temporalAnalysis =
            analysisData.framesAnalysis?.temporalAnalysis;
    } else if (mediaType === "AUDIO") {
        resultToSave.pitch = analysisData.pitch;
        resultToSave.energy = analysisData.energy;
        resultToSave.spectral = analysisData.spectral;
        resultToSave.visualization = analysisData.visualization; // Contains spectrogram info
    }

    const analysisRecord = await mediaRepository.createAnalysisResult(
        mediaId,
        resultToSave
    );

    // --- UPDATED: Handle auxiliary file uploads generically ---
    const isVideoVisualization =
        mediaType === "VIDEO" &&
        analysisData.visualizationGenerated &&
        analysisData.visualizationFilename;
    const isAudioVisualization =
        mediaType === "AUDIO" && analysisData.visualization?.spectrogramUrl;

    if (isVideoVisualization) {
        await handleAuxiliaryFileUpload(
            analysisRecord.id,
            analysisData.visualizationFilename,
            mediaId,
            userId,
            modelName,
            "video"
        );
    } else if (isAudioVisualization) {
        const filename = analysisData.visualization.spectrogramUrl
            .split("/")
            .pop();
        await handleAuxiliaryFileUpload(
            analysisRecord.id,
            filename,
            mediaId,
            userId,
            modelName,
            "image" // Spectrograms are images
        );
    }
}

// RENAMED & REFACTORED: This function is now generic.
async function handleAuxiliaryFileUpload(
    analysisId,
    filename,
    mediaId,
    userId,
    modelName,
    resourceType
) {
    const typeLabel =
        resourceType === "video" ? "visualization" : "spectrogram";

    try {
        await eventService.emitProgress({
            mediaId,
            userId,
            event: "VISUALIZATION_UPLOADING", // Keep event name for frontend
            message: `Uploading ${typeLabel} for model: ${modelName}`,
            data: { modelName },
        });

        const fileStream = await modelAnalysisService.downloadVisualization(
            filename
        );

        const uploadResponse = await storageManager.uploadStream(fileStream, {
            folder: `deepfake-${typeLabel}s`,
            resource_type: resourceType,
        });

        if (uploadResponse?.url) {
            // UPDATED: Save to the correct field based on type
            const updateData =
                resourceType === "video"
                    ? { visualizedUrl: uploadResponse.url }
                    : {
                          audioAnalysis: {
                              update: { spectrogramUrl: uploadResponse.url },
                          },
                      };

            await mediaRepository.updateAnalysis(analysisId, updateData);

            await eventService.emitProgress({
                mediaId,
                userId,
                event: "VISUALIZATION_COMPLETED", // Keep event name
                message: `${typeLabel} ready for model: ${modelName}`,
                data: { modelName, success: true, url: uploadResponse.url },
            });
        } else {
            throw new Error(
                `Storage manager did not return a URL for the ${typeLabel}.`
            );
        }
    } catch (error) {
        await eventService.emitProgress({
            mediaId,
            userId,
            event: "VISUALIZATION_COMPLETED", // Keep event name
            message: `${typeLabel} upload failed for model: ${modelName}.`,
            data: { modelName, success: false, error: error.message },
        });
        logger.error(
            `[Worker/AuxUpload] Upload failed for analysis ${analysisId}: ${error.message}`
        );
    }
}

// RENAMED: from downloadVideo to downloadMedia
async function downloadMedia(mediaUrl, originalFilename, mediaId, modelName) {
    const tempDir = path.join("temp");
    await fsPromises.mkdir(tempDir, { recursive: true });
    // Use original file extension for compatibility
    const extension = path.extname(originalFilename) || ".tmp";
    const tempFilePath = path.join(
        tempDir,
        `worker-media-${mediaId}-${modelName}-${Date.now()}${extension}`
    );

    const writer = fs.createWriteStream(tempFilePath);
    const response = await axios({
        url: mediaUrl,
        method: "GET",
        responseType: "stream",
    });
    response.data.pipe(writer);
    return new Promise((resolve, reject) => {
        writer.on("finish", () => resolve(tempFilePath));
        writer.on("error", (err) => reject(err));
    });
}

worker.on("completed", (job) =>
    logger.info(`[Worker] Job '${job.name}' (ID: ${job.id}) has completed.`)
);
worker.on("failed", (job, err) =>
    logger.error(
        `[Worker] Job '${job.name}' (ID: ${job.id}) has failed with ${err.message}`
    )
);

console.log("Media processing worker started...");
