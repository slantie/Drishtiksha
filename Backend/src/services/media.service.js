// src/services/media.service.js

import { mediaRepository } from "../repositories/media.repository.js";
import { config, mediaFlowProducer } from "../config/index.js";
import { modelAnalysisService } from "./modelAnalysis.service.js";
import storageManager from "../storage/storage.manager.js";
import { ApiError } from "../utils/ApiError.js";
import logger from "../utils/logger.js";
import { getMediaType } from "../utils/media.js";

class MediaService {
  // ... (createAndAnalyzeMedia, rerunAnalysis methods are the same)
  async createAndAnalyzeMedia(file, user, description) {
    if (!file) throw new ApiError(400, "A file is required for upload.");
    const mediaRecord = await this._createMediaRecord(file, user, description);
    await this._queueAnalysisRun(mediaRecord, 1);
    return mediaRecord;
  }

  async rerunAnalysis(mediaId, userId) {
    const media = await mediaRepository.findByIdAndUserId(mediaId, userId);
    if (!media)
      throw new ApiError(404, "Media not found or you do not have permission.");
    const latestRunNumber = await mediaRepository.findLatestRunNumber(mediaId);
    const nextRunNumber = latestRunNumber + 1;
    await this._queueAnalysisRun(media, nextRunNumber);
    return await mediaRepository.findById(mediaId);
  }

  async _createMediaRecord(file, user, description) {
    const mediaType = getMediaType(file.mimetype);
    if (!mediaType || mediaType === "UNKNOWN") {
      throw new ApiError(415, `Unsupported file type: ${file.mimetype}`);
    }

    let uploadResponse = null;
    
    try {
      // Upload file to storage
      uploadResponse = await storageManager.uploadFile(
        file.path,
        file.originalname,
        mediaType.toLowerCase() + "s"
      );

      if (!uploadResponse) {
        throw new ApiError(500, "Failed to upload file to storage.");
      }

      // Create database record - if this fails, we'll clean up the uploaded file
      const mediaRecord = await mediaRepository.create({
        filename: file.originalname,
        description,
        url: uploadResponse.url,
        publicId: uploadResponse.publicId,
        mimetype: uploadResponse.mimetype,
        size: uploadResponse.size,
        status: "QUEUED",
        userId: user.id,
        mediaType: mediaType,
      });
      
      logger.info(`[MediaService] Successfully created media record ${mediaRecord.id} for user ${user.id}`);
      return mediaRecord;
      
    } catch (error) {
      // If database creation failed but file was uploaded, clean up the file
      if (uploadResponse?.publicId) {
        logger.error(
          `[MediaService] Database creation failed after file upload. Cleaning up file: ${uploadResponse.publicId}`
        );
        try {
          await storageManager.deleteFile(uploadResponse.publicId);
          logger.info(`[MediaService] Successfully cleaned up orphaned file: ${uploadResponse.publicId}`);
        } catch (deleteError) {
          logger.error(
            `[MediaService] Failed to clean up orphaned file ${uploadResponse.publicId}: ${deleteError.message}`
          );
        }
      }
      throw error;
    }
  }

  // ... (The rest of the MediaService class remains unchanged)
  async _queueAnalysisRun(media, runNumber) {
    try {
      logger.info(
        `[MediaService] Preparing analysis run #${runNumber} for media ${media.id}`
      );
      const compatibleModels = await this._determineCompatibleModels(
        media.mediaType
      );
      const run = await mediaRepository.createAnalysisRun(media.id, runNumber);
      await mediaRepository.update(media.id, { status: "QUEUED" });

      if (compatibleModels.length === 0) {
        logger.warn(
          `No compatible models found for ${media.mediaType}. Marking run ${run.id} as analyzed.`
        );
        await mediaRepository.updateRunStatus(run.id, "ANALYZED");
        await mediaRepository.update(media.id, {
          status: "ANALYZED",
          latestAnalysisRunId: run.id,
        });
        return;
      }

      const childJobs = compatibleModels.map((modelName) => ({
        name: "run-single-analysis",
        queueName: config.MEDIA_PROCESSING_QUEUE_NAME,
        data: { mediaId: media.id, runId: run.id, modelName },
        opts: { 
          jobId: `${run.id}-${modelName}`,
          attempts: 3, // Retry failed jobs up to 3 times
          backoff: {
            type: 'exponential',
            delay: 5000, // Start with 5 second delay, then exponentially increase
          },
          removeOnComplete: {
            age: 86400, // Keep completed jobs for 24 hours
            count: 1000, // Keep last 1000 completed jobs
          },
          removeOnFail: {
            age: 604800, // Keep failed jobs for 7 days for debugging
          },
        },
      }));

      await mediaFlowProducer.add({
        name: "finalize-analysis",
        queueName: config.MEDIA_PROCESSING_QUEUE_NAME,
        data: { runId: run.id, mediaId: media.id },
        opts: { 
          jobId: `${run.id}-finalizer`,
          attempts: 2, // Retry finalizer once if it fails
          backoff: {
            type: 'fixed',
            delay: 3000, // Wait 3 seconds before retry
          },
        },
        children: childJobs,
      });
      logger.info(
        `[MediaService] Successfully queued run #${runNumber} for media ${media.id} with ${childJobs.length} models.`
      );
    } catch (error) {
      logger.error(
        `[MediaService] Failed to create analysis flow for media ${media.id}:`,
        error
      );
      await mediaRepository.update(media.id, { status: "FAILED" });
      throw new ApiError(
        500,
        "Could not queue media for analysis.",
        [],
        error.stack
      );
    }
  }

  async _determineCompatibleModels(mediaType) {
    const serverStats = await modelAnalysisService.getServerStatistics();
    console.log("Server's Sent Data: ", JSON.stringify(serverStats));
    if (!serverStats?.models_info) return [];

    return (
      serverStats.models_info
        .filter((m) => {
          if (!m.loaded) return false;
          switch (mediaType) {
            case "AUDIO":
              return m.isAudio;
            case "IMAGE":
              return m.isImage;
            case "VIDEO":
              return m.isVideo;
            default:
              return false;
          }
        })
        .map((m) => m.name) || []
    );
  }

  async getAllMediaForUser(userId) {
    return mediaRepository.findAllByUserId(userId);
  }

  async getMediaWithAnalyses(mediaId, userId) {
    const mediaItem = await mediaRepository.findByIdAndUserId(mediaId, userId);
    if (!mediaItem)
      throw new ApiError(404, "Media not found or you do not have permission.");
    return mediaItem;
  }

  async updateMedia(mediaId, userId, updateData) {
    await this.getMediaWithAnalyses(mediaId, userId); // Ensures ownership
    const allowedFields = ["description"];
    Object.keys(updateData).forEach((key) => {
      if (!allowedFields.includes(key))
        throw new ApiError(400, `Field '${key}' cannot be updated.`);
    });
    return mediaRepository.update(mediaId, updateData);
  }

  async deleteMediaById(mediaId, userId) {
    const mediaItem = await this.getMediaWithAnalyses(mediaId, userId);
    if (mediaItem.publicId) {
      // The deleteFile function in the local provider doesn't need a resourceType
      await storageManager.deleteFile(mediaItem.publicId);
    }
    await mediaRepository.deleteById(mediaId);
  }
}

export const mediaService = new MediaService();
