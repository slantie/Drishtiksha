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
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

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
  {
    connection: redisConnectionOptionsForBullMQ,
    concurrency: 1, // 🔧 FIX: Process models one-by-one to prevent memory issues
  }
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

    // Emit analysis started for this specific model
    await emitProgressEvent(
      mediaId,
      media.userId,
      "ANALYSIS_STARTED",
      `Starting ${modelName} analysis for "${media.filename}"`,
      {
        model_name: modelName,
        filename: media.filename,
        runId: runId,
        phase: "analyzing",
      }
    );

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

    // Emit analysis completed for this specific model
    await emitProgressEvent(
      mediaId,
      media.userId,
      "ANALYSIS_COMPLETED",
      `Completed ${modelName} analysis for "${media.filename}"`,
      {
        model_name: modelName,
        filename: media.filename,
        runId: runId,
        prediction: resultPayload.prediction,
        confidence: resultPayload.confidence,
        phase: "completed",
      }
    );

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
        "ANALYSIS_FAILED",
        `Analysis failed for "${media.filename}" with model ${modelName}: ${error.message}`,
        {
          model_name: modelName,
          error_message: error.message,
          filename: media.filename,
          runId: runId,
        }
      );

      // Record the error in the database
      await mediaRepository.createAnalysisError(runId, modelName, error);
    }

    // Re-throw to let BullMQ handle retries, but the error is now properly recorded
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
  let run = null;
  const MAX_RETRIES = 5;
  const RETRY_DELAY_MS = 500;

  logger.info(
    `[Finalizer] Starting finalization for run ${runId}, media ${mediaId}`
  );

  // Retry logic to handle race condition between Redis and Postgres
  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    run = await prisma.analysisRun.findUnique({
      where: { id: runId },
      include: {
        analyses: { select: { status: true, modelName: true } },
        media: { select: { userId: true, filename: true } },
      },
    });

    if (run) {
      break;
    }

    logger.warn(
      `[Finalizer] AnalysisRun ${runId} not found on attempt ${attempt}. Retrying in ${RETRY_DELAY_MS}ms...`
    );
    await wait(RETRY_DELAY_MS);
  }

  if (!run) {
    logger.error(
      `[Finalizer] Cannot finalize, AnalysisRun ${runId} not found after ${MAX_RETRIES} attempts.`
    );
    throw new Error(`AnalysisRun ${runId} not found after retries.`);
  }

  try {
    const totalAnalyses = run.analyses.length;
    const completedAnalyses = run.analyses.filter(
      (a) => a.status === "COMPLETED"
    ).length;
    const failedAnalyses = run.analyses.filter(
      (a) => a.status === "FAILED"
    ).length;
    const pendingAnalyses = totalAnalyses - completedAnalyses - failedAnalyses;

    logger.info(
      `[Finalizer] Run ${runId} status: ${completedAnalyses} completed, ${failedAnalyses} failed, ${pendingAnalyses} pending out of ${totalAnalyses} models.`
    );

    let finalStatus;
    // Improved status determination logic
    if (completedAnalyses > 0 && pendingAnalyses === 0) {
      // All jobs are done (either completed or failed), and at least one succeeded.
      finalStatus = "ANALYZED";
    } else if (completedAnalyses === 0 && pendingAnalyses === 0) {
      // All jobs are done, and none succeeded.
      finalStatus = "FAILED";
    } else if (completedAnalyses > 0 && pendingAnalyses > 0) {
      // Some succeeded, some still pending - partially analyzed state
      finalStatus = "PARTIALLY_ANALYZED";
    } else {
      // Still processing, finalizer might have run prematurely
      finalStatus = "PROCESSING";
    }

    // Update both run and media status in a transaction for consistency
    await prisma.$transaction(async (tx) => {
      await tx.analysisRun.update({
        where: { id: run.id },
        data: { status: finalStatus },
      });

      await tx.media.update({
        where: { id: mediaId },
        data: {
          status: finalStatus,
          latestAnalysisRunId: run.id,
        },
      });
    });

    logger.info(
      `[Finalizer] ✅ Updated database: Run ${runId} and Media ${mediaId} status set to ${finalStatus}`
    );

    // Emit final status event
    if (run.media) {
      await emitProgressEvent(
        mediaId,
        run.media.userId,
        finalStatus === "ANALYZED" ? "ANALYSIS_COMPLETE" : "ANALYSIS_FAILED",
        finalStatus === "ANALYZED"
          ? `Analysis completed for "${run.media.filename}" (${completedAnalyses}/${totalAnalyses} models succeeded)`
          : failedAnalyses === totalAnalyses
          ? `All ${totalAnalyses} models failed for "${run.media.filename}"`
          : `Analysis partially failed for "${run.media.filename}" (${completedAnalyses}/${totalAnalyses} models succeeded)`,
        {
          runId: run.id,
          totalAnalyses,
          completedAnalyses,
          failedAnalyses,
          pendingAnalyses,
          finalStatus,
          filename: run.media.filename,
        }
      );
    }

    logger.info(
      `[Finalizer] ✅ Finalized AnalysisRun ${run.id} for media ${mediaId} with status: ${finalStatus} (${completedAnalyses}/${totalAnalyses} completed, ${failedAnalyses} failed)`
    );

    return {
      runId,
      mediaId,
      finalStatus,
      completedAnalyses,
      failedAnalyses,
      totalAnalyses,
    };
  } catch (error) {
    logger.error(
      `[Finalizer] ❌ Error finalizing AnalysisRun ${runId}: ${error.message}`,
      { stack: error.stack }
    );

    // Ensure media status is set to FAILED even if finalization fails
    try {
      await mediaRepository.update(mediaId, { status: "FAILED" });
      logger.info(
        `[Finalizer] Set media ${mediaId} status to FAILED after finalization error`
      );
    } catch (updateError) {
      logger.error(
        `[Finalizer] Failed to update media ${mediaId} status to FAILED: ${updateError.message}`
      );
    }

    throw error;
  }
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

// Graceful shutdown handler
const gracefulShutdown = async (signal) => {
  logger.info(`\n${signal} received. Shutting down worker gracefully...`);

  try {
    // Stop accepting new jobs
    await worker.close();
    logger.info("[Worker] Stopped accepting new jobs.");

    // Close Redis connections
    await redisPublisher.quit();
    logger.info("[Worker] Closed Redis publisher connection.");

    // Close Prisma connection
    await prisma.$disconnect();
    logger.info("[Worker] Closed Prisma database connection.");

    logger.info("✅ Worker shutdown complete.");
    process.exit(0);
  } catch (error) {
    logger.error(`[Worker] Error during shutdown: ${error.message}`);
    process.exit(1);
  }
};

// Set up signal handlers for graceful shutdown
process.on("SIGTERM", () => gracefulShutdown("SIGTERM"));
process.on("SIGINT", () => gracefulShutdown("SIGINT"));

// Handle uncaught errors
process.on("uncaughtException", (error) => {
  logger.error("[Worker] Uncaught Exception:", error);
  gracefulShutdown("UNCAUGHT_EXCEPTION");
});

process.on("unhandledRejection", (reason, promise) => {
  logger.error("[Worker] Unhandled Rejection at:", promise, "reason:", reason);
  gracefulShutdown("UNHANDLED_REJECTION");
});

logger.info("🚀 Media processing worker started and is listening for jobs.");
