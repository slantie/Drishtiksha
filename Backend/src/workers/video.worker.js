// src/workers/video.worker.js

import { Worker } from "bullmq";
import { promises as fs } from "fs";
import path from "path";
import axios from "axios";
import dotenv from "dotenv";
import { videoRepository } from "../repositories/video.repository.js";
import { modelAnalysisService } from "../services/modelAnalysis.service.js";
import { videoService } from "../services/video.service.js";
import { VIDEO_PROCESSING_QUEUE_NAME } from "../config/constants.js";
import logger from "../utils/logger.js";
import { uploadStreamToCloudinary } from "../utils/cloudinary.js";

dotenv.config({ path: "./.env" });

const redisConnection = {
    host: process.env.REDIS_URL
        ? new URL(process.env.REDIS_URL).hostname
        : "localhost",
    port: process.env.REDIS_URL
        ? parseInt(new URL(process.env.REDIS_URL).port)
        : 6379,
};

// REASON: This defines the worker process that will handle jobs from the queue.
// It runs independently of the main API server, making the system scalable and resilient.
const worker = new Worker(
    VIDEO_PROCESSING_QUEUE_NAME,
    async (job) => {
        const { videoId } = job.data;
        logger.info(`[Worker] Picked up job ${job.id} for video ${videoId}`);

        const video = await videoRepository.findById(videoId);
        if (!video) {
            throw new Error(`[Worker] Video ${videoId} not found in database.`);
        }

        // This is a placeholder for the real io instance.
        // In a scaled environment, you would use a Redis pub/sub or another mechanism to signal the API server.
        // For a single server, this approach requires the worker to be aware of the io instance,
        // which is complex. A better approach is to have the worker update the DB,
        // and a separate mechanism triggers the socket emission.
        // For simplicity here, we'll just log it. A proper implementation needs a pub/sub.
        const mockIo = {
            to: () => ({
                emit: (event, data) =>
                    logger.info(
                        `[Worker] Mock emit: ${event} for user ${video.userId}`
                    ),
            }),
        };

        await videoRepository.updateStatus(videoId, "PROCESSING");
        videoService.emitVideoUpdate(mockIo, video.userId, {
            ...video,
            status: "PROCESSING",
        });

        let localVideoPath;
        let analysisSuccessCount = 0;
        let totalAnalysesAttempted = 0;

        try {
            const serverStats =
                await modelAnalysisService.getServerStatistics();
            await videoRepository.storeServerHealth(serverStats);

            const availableModels =
                serverStats.models_info
                    ?.filter((m) => m.loaded)
                    .map((m) => m.name) || [];
            totalAnalysesAttempted = availableModels.length;

            if (totalAnalysesAttempted === 0)
                throw new Error(
                    "No models are available on the analysis server."
                );

            localVideoPath = await downloadVideo(video.url, videoId);

            for (const modelName of availableModels) {
                const success = await runAndSaveComprehensiveAnalysis(
                    videoId,
                    localVideoPath,
                    modelName,
                    serverStats
                );
                if (success) analysisSuccessCount++;
            }

            let finalStatus = "FAILED";
            if (
                analysisSuccessCount === totalAnalysesAttempted &&
                totalAnalysesAttempted > 0
            )
                finalStatus = "ANALYZED";
            else if (analysisSuccessCount > 0)
                finalStatus = "PARTIALLY_ANALYZED";

            await videoRepository.updateStatus(videoId, finalStatus);
            const finalVideoState = await videoRepository.findById(videoId);
            videoService.emitVideoUpdate(mockIo, video.userId, finalVideoState);

            logger.info(
                `[Worker] ✅ Job ${job.id} completed. Status: ${finalStatus}. [${analysisSuccessCount}/${totalAnalysesAttempted} successful]`
            );
        } catch (error) {
            logger.error(
                `[Worker] ❌ Job ${job.id} for video ${videoId} failed: ${error.message}`
            );
            await videoRepository.updateStatus(videoId, "FAILED");
            const failedVideoState = await videoRepository.findById(videoId);
            videoService.emitVideoUpdate(
                mockIo,
                video.userId,
                failedVideoState
            );
            throw error; // Throw error to let BullMQ handle retries
        } finally {
            if (localVideoPath)
                await fs
                    .unlink(localVideoPath)
                    .catch((err) =>
                        logger.error(`[Worker] Cleanup failed: ${err.message}`)
                    );
        }
    },
    { connection: redisConnection }
);

// Helper functions now live inside the worker file
async function runAndSaveComprehensiveAnalysis(
    videoId,
    videoPath,
    modelName,
    serverStats
) {
    const modelEnum = modelAnalysisService.mapModelNameToEnum(modelName);
    try {
        const response = await modelAnalysisService.analyzeVideoComprehensive(
            videoPath,
            modelName,
            videoId
        );
        const { modelInfo, systemInfo } =
            modelAnalysisService.mapServerStatsToDbSchema(
                serverStats,
                modelName
            );
        const resultToSave = {
            ...response.data,
            model: modelEnum,
            modelVersion: response.model_used,
            analysisType: "COMPREHENSIVE",
            modelInfo,
            systemInfo,
        };
        const analysisRecord = await videoRepository.createAnalysisResult(
            videoId,
            resultToSave
        );
        if (
            response.data.visualization_generated &&
            response.data.visualization_filename
        ) {
            await handleVisualizationUpload(
                analysisRecord.id,
                response.data.visualization_filename
            );
        }
        return true;
    } catch (error) {
        logger.error(
            `[Worker/Analysis] ${modelName} failed for ${videoId}: ${error.message}`
        );
        await videoRepository.createAnalysisError(
            videoId,
            modelEnum,
            "COMPREHENSIVE",
            error
        );
        return false;
    }
}

async function handleVisualizationUpload(analysisId, filename) {
    try {
        const videoStream = await modelAnalysisService.downloadVisualization(
            filename
        );
        const cloudinaryResponse = await uploadStreamToCloudinary(videoStream, {
            folder: "deepfake-visualizations",
            resource_type: "video",
        });
        if (cloudinaryResponse && cloudinaryResponse.secure_url) {
            await videoRepository.updateAnalysis(analysisId, {
                visualizedUrl: cloudinaryResponse.secure_url,
            });
        }
    } catch (error) {
        logger.error(
            `[Worker/Viz] Upload failed for analysis ${analysisId}: ${error.message}`
        );
    }
}

async function downloadVideo(videoUrl, videoId) {
    const tempDir = path.join("temp");
    await fs.mkdir(tempDir, { recursive: true });
    const tempFilePath = path.join(
        tempDir,
        `worker-video-${videoId}-${Date.now()}.mp4`
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
    logger.info(`[Worker] Job ${job.id} has completed.`)
);
worker.on("failed", (job, err) =>
    logger.error(`[Worker] Job ${job.id} has failed with ${err.message}`)
);

console.log("Video processing worker started...");
