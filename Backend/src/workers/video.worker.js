// src/workers/video.worker.js
import { Worker } from "bullmq";
import fs from "fs";
import { promises as fsPromises } from "fs";
import path from "path";
import axios from "axios";
import dotenv from "dotenv";
import { videoRepository } from "../repositories/video.repository.js";
import { modelAnalysisService } from "../services/modelAnalysis.service.js";
import { redisConnection } from "../config/queue.js";
import { VIDEO_PROCESSING_QUEUE_NAME } from "../config/constants.js";
import logger from "../utils/logger.js";
import { eventService } from "../services/event.service.js";
import storageManager from "../storage/storage.manager.js";
import { toCamelCase } from "../utils/formatKeys.js";

dotenv.config({ path: "./.env" });

const worker = new Worker(
    VIDEO_PROCESSING_QUEUE_NAME,
    async (job) => {
        logger.info(`[Worker] Picked up job '${job.name}' (ID: ${job.id})`);
        switch (job.name) {
            // --- REFACTORED: Removed 'analysis-flow' handler ---
            case "run-single-analysis":
                return await handleSingleAnalysis(job);
            case "finalize-analysis":
                return await handleFinalizeAnalysis(job);
            // --- END REFACTOR ---
            default:
                throw new Error(`Unknown job name: ${job.name}`);
        }
    },
    { connection: redisConnection, concurrency: 5 }
);

async function handleSingleAnalysis(job) {
    const { videoId, modelName, serverStats } = job.data;
    let localVideoPath;
    let userId;
    let isTempFile = false;

    try {
        const video = await videoRepository.findById(videoId);
        if (!video) throw new Error(`Video ${videoId} not found.`);
        userId = video.userId;

        await videoRepository.updateStatus(videoId, "PROCESSING");
        await eventService.emitProgress({
            videoId,
            userId,
            event: "ANALYSIS_STARTED",
            message: `Analysis started for model: ${modelName}`,
            data: { modelName },
        });

        // --- MODIFIED: Get video path based on the storage provider ---
        if (process.env.STORAGE_PROVIDER === "local") {
            // For local storage, construct the absolute path from the relative path stored in publicId
            const localStoragePath =
                process.env.LOCAL_STORAGE_PATH || "public/media";
            localVideoPath = path.resolve(localStoragePath, video.publicId);
            logger.info(
                `[Worker] Using local file for analysis: ${localVideoPath}`
            );
            if (!fs.existsSync(localVideoPath)) {
                throw new Error(
                    `Local video file not found at: ${localVideoPath}`
                );
            }
        } else {
            // For Cloudinary, download the video to a temporary file
            localVideoPath = await downloadVideo(video.url, videoId, modelName);
            isTempFile = true; // Mark this file for cleanup
            logger.info(
                `[Worker] Downloaded Cloudinary video to temp path: ${localVideoPath}`
            );
        }

        await runAndSaveComprehensiveAnalysis(
            videoId,
            userId,
            localVideoPath,
            modelName,
            serverStats
        );

        await eventService.emitProgress({
            videoId,
            userId,
            event: "ANALYSIS_COMPLETED",
            message: `Analysis completed for model: ${modelName}`,
            data: { modelName, success: true },
        });
    } catch (error) {
        logger.error(
            `[Worker/Analysis] Job for model ${modelName} on video ${videoId} failed: ${error.message}`
        );
        await videoRepository.createAnalysisError(
            videoId,
            modelName,
            "COMPREHENSIVE",
            error
        );
        if (userId) {
            await eventService.emitProgress({
                videoId,
                userId,
                event: "ANALYSIS_COMPLETED",
                message: `Analysis failed for model: ${modelName}.`,
                data: { modelName, success: false, error: error.message },
            });
        }
        throw error;
    } finally {
        if (localVideoPath && isTempFile) {
            await fsPromises
                .unlink(localVideoPath)
                .catch((err) =>
                    logger.error(
                        `[Worker] Cleanup failed for temp file ${localVideoPath}: ${err.message}`
                    )
                );
        }
    }
}

async function handleFinalizeAnalysis(job) {
    const { videoId, totalAnalysesAttempted } = job.data;
    const video = await videoRepository.findById(videoId);
    if (!video) throw new Error(`Cannot finalize, video ${videoId} not found.`);

    // Check both completed and failed analyses to get a true count of attempts
    const completedAnalyses = video.analyses.filter(
        (a) => a.status === "COMPLETED"
    ).length;
    const failedAnalyses = video.analyses.filter(
        (a) => a.status === "FAILED"
    ).length;
    const totalProcessed = completedAnalyses + failedAnalyses;

    let finalStatus = "FAILED";
    if (
        completedAnalyses === totalAnalysesAttempted &&
        totalAnalysesAttempted > 0
    ) {
        finalStatus = "ANALYZED";
    } else if (completedAnalyses > 0) {
        finalStatus = "PARTIALLY_ANALYZED";
    }

    await videoRepository.updateStatus(videoId, finalStatus);
    logger.info(
        `[Finalizer] Finalized video ${videoId} with status: ${finalStatus} [${completedAnalyses}/${totalAnalysesAttempted} successful]`
    );
}

async function runAndSaveComprehensiveAnalysis(
    videoId,
    userId,
    videoPath,
    modelName,
    serverStats
) {
    const response = await modelAnalysisService.analyzeVideoComprehensive(
        videoPath,
        modelName,
        videoId,
        userId
    );

    const analysisData = toCamelCase(response.data);
    const modelUsed = response.model_used;

    const { modelInfo, systemInfo } =
        modelAnalysisService.mapServerStatsToDbSchema(serverStats, modelName);

    const resultToSave = {
        prediction: analysisData.prediction,
        confidence: analysisData.confidence,
        processingTime: analysisData.processingTime,
        metrics: analysisData.metrics,
        framePredictions: analysisData.framesAnalysis?.framePredictions,
        temporalAnalysis: analysisData.framesAnalysis?.temporalAnalysis,
        model: modelName,
        modelVersion: modelUsed,
        analysisType: "COMPREHENSIVE",
        modelInfo,
        systemInfo,
    };

    const analysisRecord = await videoRepository.createAnalysisResult(
        videoId,
        resultToSave
    );

    if (
        analysisData.visualizationGenerated &&
        analysisData.visualizationFilename
    ) {
        await handleVisualizationUpload(
            analysisRecord.id,
            analysisData.visualizationFilename,
            videoId,
            userId,
            modelName
        );
    }
}

async function handleVisualizationUpload(
    analysisId,
    filename,
    videoId,
    userId,
    modelName
) {
    try {
        await eventService.emitProgress({
            videoId,
            userId,
            event: "VISUALIZATION_UPLOADING",
            message: `Uploading visualization for model: ${modelName}`,
            data: { modelName },
        });

        const videoStream = await modelAnalysisService.downloadVisualization(
            filename
        );

        // --- MODIFIED: Use the agnostic storage manager to upload the stream ---
        const uploadResponse = await storageManager.uploadStream(videoStream, {
            folder: "deepfake-visualizations", // Used by both providers
            resource_type: "video", // Used by Cloudinary, ignored by local
        });

        if (uploadResponse?.url) {
            await videoRepository.updateAnalysis(analysisId, {
                visualizedUrl: uploadResponse.url,
            });
            await eventService.emitProgress({
                videoId,
                userId,
                event: "VISUALIZATION_COMPLETED",
                message: `Visualization ready for model: ${modelName}`,
                data: {
                    modelName,
                    success: true,
                    url: uploadResponse.url,
                },
            });
        } else {
            throw new Error(
                "Storage manager did not return a URL for the visualization."
            );
        }
    } catch (error) {
        await eventService.emitProgress({
            videoId,
            userId,
            event: "VISUALIZATION_COMPLETED",
            message: `Visualization failed for model: ${modelName}.`,
            data: { modelName, success: false, error: error.message },
        });
        logger.error(
            `[Worker/Viz] Upload failed for analysis ${analysisId}: ${error.message}`
        );
    }
}

async function downloadVideo(videoUrl, videoId, modelName) {
    const tempDir = path.join("temp");
    await fsPromises.mkdir(tempDir, { recursive: true });
    const tempFilePath = path.join(
        tempDir,
        `worker-video-${videoId}-${modelName}-${Date.now()}.mp4`
    );
    const writer = fs.createWriteStream(tempFilePath);
    const response = await axios({
        url: videoUrl,
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
console.log("Video processing worker started...");
