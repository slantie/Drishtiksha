// src/workers/media.worker.js

import { Worker } from "bullmq";
import { promises as fsPromises, createWriteStream, existsSync } from "fs";
import path from "path";
import axios from "axios";
import { fileURLToPath } from "url";

import { mediaRepository } from "../repositories/media.repository.js";
import { modelAnalysisService } from "../services/modelAnalysis.service.js";
import {
  config,
  prisma,
  redisConnectionOptionsForBullMQ,
} from "../config/index.js";
import { redisPublisher } from "../config/redis.js";
import logger from "../utils/logger.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const worker = new Worker(
  config.MEDIA_PROCESSING_QUEUE_NAME,
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
  { connection: redisConnectionOptionsForBullMQ, concurrency: 5 }
);

// Helper function to emit custom progress events to Redis
const emitProgressEvent = async (
  mediaId,
  userId,
  event,
  message,
  data = {}
) => {
  try {
    const progressEvent = {
      media_id: mediaId,
      user_id: userId,
      event: event,
      message: message,
      data: {
        ...data,
        timestamp: new Date().toISOString(),
      },
    };

    await redisPublisher.publish(
      config.MEDIA_PROGRESS_CHANNEL_NAME,
      JSON.stringify(progressEvent)
    );
    logger.debug(`[Worker] Emitted '${event}' event for media ${mediaId}`);
  } catch (error) {
    logger.error(
      `[Worker] Failed to emit progress event '${event}' for media ${mediaId}:`,
      error
    );
  }
};

async function handleSingleAnalysis(job) {
  const { mediaId, runId, modelName } = job.data;
  let localMediaPath;
  let isTempFile = false;
  let media;

  try {
    media = await mediaRepository.findById(mediaId);
    if (!media) {
      logger.warn(`[Worker] Media ${mediaId} not found, skipping job.`);
      return { status: "skipped", reason: "Media record not found." };
    }

    await mediaRepository.updateRunStatus(runId, "PROCESSING");
    await mediaRepository.update(mediaId, { status: "PROCESSING" });

    // Emit processing started event for this media
    await emitProgressEvent(
      mediaId,
      media.userId,
      "PROCESSING_STARTED",
      `Starting analysis of "${media.filename}"`,
      {
        model_name: "Backend Worker",
        filename: media.filename,
        mediaType: media.mediaType,
        runId: runId,
      }
    );

    localMediaPath = await _getLocalMediaPath(media);
    isTempFile = config.STORAGE_PROVIDER !== "local";

    const resultPayload = await modelAnalysisService.runAnalysis(
      localMediaPath,
      modelName,
      mediaId,
      media.userId
    );

    await mediaRepository.createAnalysisResult(runId, {
      modelName,
      prediction: resultPayload.prediction,
      confidence: resultPayload.confidence,
      resultPayload,
    });

    logger.info(
      `[Worker] ✅ Successfully completed ${modelName} analysis for media ${mediaId}`
    );
  } catch (error) {
    logger.error(
      `[Worker] Analysis job for ${modelName} on media ${mediaId} failed: ${error.message}`,
      { stack: error.stack }
    );

    // Emit processing failed event
    if (media) {
      await emitProgressEvent(
        mediaId,
        media.userId,
        "PROCESSING_FAILED",
        `Analysis failed for "${media.filename}" with model ${modelName}`,
        {
          model_name: modelName,
          error_message: error.message,
          filename: media.filename,
          runId: runId,
        }
      );
      await mediaRepository.createAnalysisError(runId, modelName, error);
    }
    throw error;
  } finally {
    if (localMediaPath && isTempFile) {
      await fsPromises
        .unlink(localMediaPath)
        .catch((err) =>
          logger.error(
            `[Worker] Failed to clean up temp file ${localMediaPath}: ${err.message}`
          )
        );
    }
  }
}

async function handleFinalizeAnalysis(job) {
  const { runId, mediaId } = job.data;
  const run = await prisma.analysisRun.findUnique({
    where: { id: runId },
    include: { analyses: { select: { status: true } } },
  });

  if (!run) {
    logger.error(
      `[Finalizer] Cannot finalize, AnalysisRun ${runId} not found.`
    );
    return { status: "error", reason: "AnalysisRun not found." };
  }

  const failedAnalyses = run.analyses.filter(
    (a) => a.status === "FAILED"
  ).length;
  const finalStatus = failedAnalyses > 0 ? "FAILED" : "ANALYZED";

  await mediaRepository.updateRunStatus(run.id, finalStatus);
  await mediaRepository.update(mediaId, {
    status: finalStatus,
    latestAnalysisRunId: run.id,
  });

  logger.info(
    `[Finalizer] Finalized AnalysisRun ${run.id} for media ${mediaId} with status: ${finalStatus}`
  );
  return { runId, mediaId, finalStatus };
}

async function _getLocalMediaPath(media) {
  const projectRoot = path.resolve(__dirname, "..", "..");
  if (config.STORAGE_PROVIDER === "local") {
    const localPath = path.join(
      projectRoot,
      config.LOCAL_STORAGE_PATH,
      media.publicId
    );
    if (!existsSync(localPath))
      throw new Error(`Local file not found at: ${localPath}`);
    return localPath;
  } else {
    const tempDir = path.join(projectRoot, "temp", "worker-downloads");
    await fsPromises.mkdir(tempDir, { recursive: true });
    const extension = path.extname(media.filename) || ".tmp";
    const tempFilePath = path.join(
      tempDir,
      `${media.id}-${Date.now()}${extension}`
    );
    const writer = createWriteStream(tempFilePath);
    const response = await axios({
      url: media.url,
      method: "GET",
      responseType: "stream",
    });
    response.data.pipe(writer);
    return new Promise((resolve, reject) => {
      writer.on("finish", () => resolve(tempFilePath));
      writer.on("error", (err) => reject(err));
    });
  }
}

worker.on("completed", (job) =>
  logger.info(`[Worker] Job '${job.name}' (ID: ${job.id}) has completed.`)
);
worker.on("failed", (job, err) =>
  logger.error(
    `[Worker] Job '${job.name}' (ID: ${job.id}) has failed: ${err.message}`
  )
);
logger.info("🚀 Media processing worker started and is listening for jobs.");
