// src/workers/video.worker.js
import { Worker } from "bullmq";
import fs from "fs";
import { promises as fsPromises } from "fs";
import path from "path";
import axios from "axios";
import dotenv from "dotenv";
import { videoRepository } from "../repositories/video.repository.js";
import { modelAnalysisService } from "../services/modelAnalysis.service.js";
import { eventService } from "../services/event.service.js";
import { VIDEO_PROCESSING_QUEUE_NAME } from "../config/constants.js";
import { videoFlowProducer } from "../config/queue.js";
import logger from "../utils/logger.js";
import { uploadStreamToCloudinary } from "../utils/cloudinary.js";
import { toCamelCase } from "../utils/formatKeys.js";

dotenv.config({ path: "./.env" });

const redisConnection = {
    host: process.env.REDIS_URL
        ? new URL(process.env.REDIS_URL).hostname
        : "localhost",
    port: process.env.REDIS_URL
        ? parseInt(new URL(process.env.REDIS_URL).port)
        : 6379,
};

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

// async function handleAnalysisFlow(job) {
//     const { videoId } = job.data;
//     const video = await videoRepository.findById(videoId);
//     if (!video) throw new Error(`Video ${videoId} not found.`);

//     const existingFlow = await videoFlowProducer.getFlow({
//         id: videoId,
//         queueName: VIDEO_PROCESSING_QUEUE_NAME,
//     });

//     if (existingFlow?.children?.length > 0) {
//         logger.warn(
//             `[Flow] Analysis flow for video ${videoId} already exists. Skipping creation.`
//         );
//         return;
//     }

//     await eventService.emitProgress({
//         videoId,
//         userId: video.userId,
//         event: "PROCESSING_STARTED",
//         message: "Your video is now being processed.",
//     });

//     await videoRepository.updateStatus(videoId, "PROCESSING");
//     const serverStats = await modelAnalysisService.getServerStatistics();
//     const availableModels =
//         serverStats.models_info?.filter((m) => m.loaded).map((m) => m.name) ||
//         [];

//     if (availableModels.length === 0) {
//         throw new Error("No models are available on the analysis server.");
//     }

//     const childJobs = availableModels.map((modelName) => ({
//         name: "run-single-analysis",
//         data: { videoId, modelName, serverStats },
//         queueName: VIDEO_PROCESSING_QUEUE_NAME,
//         opts: { jobId: `${videoId}-${modelName}` },
//     }));

//     await videoFlowProducer.add({
//         name: "finalize-analysis",
//         queueName: VIDEO_PROCESSING_QUEUE_NAME,
//         data: { videoId, totalAnalysesAttempted: childJobs.length },
//         opts: { jobId: `${videoId}-finalizer` },
//         children: childJobs,
//     });

//     logger.info(
//         `[Flow] Created analysis flow for video ${videoId} with ${childJobs.length} child jobs.`
//     );
// }

async function handleSingleAnalysis(job) {
    const { videoId, modelName, serverStats } = job.data;
    let localVideoPath;
    let userId;

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

        localVideoPath = await downloadVideo(video.url, videoId, modelName);
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
        if (localVideoPath) {
            await fsPromises
                .unlink(localVideoPath)
                .catch((err) =>
                    logger.error(
                        `[Worker] Cleanup failed for ${localVideoPath}: ${err.message}`
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
        const cloudinaryResponse = await uploadStreamToCloudinary(videoStream, {
            folder: "deepfake-visualizations",
            resource_type: "video",
        });
        if (cloudinaryResponse?.secure_url) {
            await videoRepository.updateAnalysis(analysisId, {
                visualizedUrl: cloudinaryResponse.secure_url,
            });
            await eventService.emitProgress({
                videoId,
                userId,
                event: "VISUALIZATION_COMPLETED",
                message: `Visualization ready for model: ${modelName}`,
                data: {
                    modelName,
                    success: true,
                    url: cloudinaryResponse.secure_url,
                },
            });
        } else {
            throw new Error("Cloudinary upload did not return a secure URL.");
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
