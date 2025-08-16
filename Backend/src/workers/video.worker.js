// src/workers/video.worker.js

import { Worker } from "bullmq";
import fs from "fs";
import { promises as fsPromises } from "fs";
import path from "path";
import axios from "axios";
import dotenv from "dotenv";
import { videoRepository } from "../repositories/video.repository.js";
import { modelAnalysisService } from "../services/modelAnalysis.service.js";
import { VIDEO_PROCESSING_QUEUE_NAME } from "../config/constants.js";
import { videoFlowProducer } from "../config/queue.js";
import logger from "../utils/logger.js";
import { uploadStreamToCloudinary } from "../utils/cloudinary.js";

dotenv.config({ path: "./.env" });

const redisConnection = {
    host: process.env.REDIS_URL ? new URL(process.env.REDIS_URL).hostname : "localhost",
    port: process.env.REDIS_URL ? parseInt(new URL(process.env.REDIS_URL).port) : 6379,
};

const worker = new Worker(VIDEO_PROCESSING_QUEUE_NAME, async (job) => {
    logger.info(`[Worker] Picked up job '${job.name}' (ID: ${job.id})`);
    switch (job.name) {
        case 'analysis-flow': return await handleAnalysisFlow(job);
        case 'run-single-analysis': return await handleSingleAnalysis(job);
        case 'finalize-analysis': return await handleFinalizeAnalysis(job);
        default: throw new Error(`Unknown job name: ${job.name}`);
    }
}, { connection: redisConnection, concurrency: 5 });


async function handleAnalysisFlow(job) {
    const { videoId } = job.data;
    
    // CORRECTED: Added a null check to prevent the crash.
    // REASON: getFlow returns undefined if no flow exists. This is the definitive fix for the 'cannot read properties of undefined' error.
    const existingFlow = await videoFlowProducer.getFlow({
        id: videoId,
        queueName: VIDEO_PROCESSING_QUEUE_NAME,
    });

    if (existingFlow?.children?.length > 0) {
        logger.warn(`[Flow] Analysis flow for video ${videoId} already exists. Skipping creation.`);
        return;
    }

    await videoRepository.updateStatus(videoId, "PROCESSING");
    const serverStats = await modelAnalysisService.getServerStatistics();
    const availableModels = serverStats.models_info?.filter(m => m.loaded).map(m => m.name) || [];

    if (availableModels.length === 0) {
        throw new Error("No models are available on the analysis server.");
    }

    const childJobs = availableModels.map(modelName => ({
        name: 'run-single-analysis',
        data: { videoId, modelName, serverStats },
        queueName: VIDEO_PROCESSING_QUEUE_NAME,
        opts: { jobId: `${videoId}-${modelName}` }
    }));
    
    // IMPORTANT: The parent job's ID is the videoId. This is key for the event listener.
    await videoFlowProducer.add({
        name: 'finalize-analysis',
        queueName: VIDEO_PROCESSING_QUEUE_NAME,
        data: { videoId, totalAnalysesAttempted: childJobs.length },
        opts: { jobId: `${videoId}-finalizer` },
        children: childJobs,
    });
    
    logger.info(`[Flow] Created analysis flow for video ${videoId} with ${childJobs.length} child jobs.`);
}


// ... (rest of the file is unchanged, collapsed for brevity)
async function handleSingleAnalysis(job) {
    const { videoId, modelName, serverStats } = job.data;
    let localVideoPath;
    try {
        const video = await videoRepository.findById(videoId);
        if (!video) throw new Error(`Video ${videoId} not found.`);
        localVideoPath = await downloadVideo(video.url, videoId, modelName);
        await runAndSaveComprehensiveAnalysis(videoId, localVideoPath, modelName, serverStats);
    } finally {
        if (localVideoPath) await fsPromises.unlink(localVideoPath).catch(err => logger.error(`[Worker] Cleanup failed for ${localVideoPath}: ${err.message}`));
    }
}
async function handleFinalizeAnalysis(job) {
    const { videoId, totalAnalysesAttempted } = job.data;
    const video = await videoRepository.findById(videoId);
    if (!video) throw new Error(`Cannot finalize, video ${videoId} not found.`);
    const successfulAnalyses = video.analyses.filter(a => a.status === "COMPLETED").length;
    let finalStatus = "FAILED";
    if (successfulAnalyses === totalAnalysesAttempted && totalAnalysesAttempted > 0) finalStatus = "ANALYZED";
    else if (successfulAnalyses > 0) finalStatus = "PARTIALLY_ANALYZED";
    await videoRepository.updateStatus(videoId, finalStatus);
    logger.info(`[Finalizer] Finalized video ${videoId} with status: ${finalStatus} [${successfulAnalyses}/${totalAnalysesAttempted} successful]`);
}
async function runAndSaveComprehensiveAnalysis(videoId, videoPath, modelName, serverStats) {
    try {
        const response = await modelAnalysisService.analyzeVideoComprehensive(videoPath, modelName, videoId);
        const analysisData = response.data;
        const { modelInfo, systemInfo } = modelAnalysisService.mapServerStatsToDbSchema(serverStats, modelName);
        const resultToSave = {
            prediction: analysisData.prediction, confidence: analysisData.confidence, processingTime: analysisData.processing_time, metrics: analysisData.metrics, framePredictions: analysisData.frames_analysis?.frame_predictions, temporalAnalysis: analysisData.frames_analysis?.temporal_analysis, model: modelName, modelVersion: response.model_used, analysisType: "COMPREHENSIVE", modelInfo, systemInfo,
        };
        const analysisRecord = await videoRepository.createAnalysisResult(videoId, resultToSave);
        if (analysisData.visualization_generated && analysisData.visualization_filename) {
            await handleVisualizationUpload(analysisRecord.id, analysisData.visualization_filename);
        }
    } catch (error) {
        logger.error(`[Worker/Analysis] ${modelName} failed for ${videoId}: ${error.message}`);
        await videoRepository.createAnalysisError(videoId, modelName, "COMPREHENSIVE", error);
        throw error;
    }
}
async function handleVisualizationUpload(analysisId, filename) {
    try {
        const videoStream = await modelAnalysisService.downloadVisualization(filename);
        const cloudinaryResponse = await uploadStreamToCloudinary(videoStream, { folder: "deepfake-visualizations", resource_type: "video" });
        if (cloudinaryResponse?.secure_url) {
            await videoRepository.updateAnalysis(analysisId, { visualizedUrl: cloudinaryResponse.secure_url });
        }
    } catch (error) {
        logger.error(`[Worker/Viz] Upload failed for analysis ${analysisId}: ${error.message}`);
    }
}
async function downloadVideo(videoUrl, videoId, modelName) {
    const tempDir = path.join("temp");
    await fsPromises.mkdir(tempDir, { recursive: true });
    const tempFilePath = path.join(tempDir,`worker-video-${videoId}-${modelName}-${Date.now()}.mp4`);
    const writer = fs.createWriteStream(tempFilePath);
    const response = await axios({ url: videoUrl, method: "GET", responseType: "stream" });
    response.data.pipe(writer);
    return new Promise((resolve, reject) => {
        writer.on("finish", () => resolve(tempFilePath));
        writer.on("error", (err) => reject(err));
    });
}
worker.on("completed", job => logger.info(`[Worker] Job '${job.name}' (ID: ${job.id}) has completed.`));
worker.on("failed", (job, err) => logger.error(`[Worker] Job '${job.name}' (ID: ${job.id}) has failed with ${err.message}`));
console.log("Video processing worker started...");