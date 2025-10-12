// src/services/media.service.js

import { mediaRepository } from "../repositories/media.repository.js";
import { config, mediaFlowProducer } from "../config/index.js";
import { modelAnalysisService } from "./modelAnalysis.service.js";
import storageManager from "../storage/storage.manager.js";
import { ApiError } from "../utils/ApiError.js";
import logger from "../utils/logger.js";
import { getMediaType } from "../utils/media.js";
import { redisPublisher } from "../config/redis.js";

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

      // Extract metadata from the uploaded file
      let hasAudio = null;
      let metadata = null;
      
      if (uploadResponse.metadata) {
        metadata = uploadResponse.metadata;
        // For videos, check if audio track exists
        if (mediaType === "VIDEO" && metadata.audio) {
          hasAudio = true;
        } else if (mediaType === "VIDEO" && !metadata.audio) {
          hasAudio = false;
        }
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
        hasAudio: hasAudio,
        metadata: metadata,
      });
      
      logger.info(`[MediaService] Successfully created media record ${mediaRecord.id} for user ${user.id}${hasAudio !== null ? ` (hasAudio: ${hasAudio})` : ''}`);
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
        media.mediaType,
        media.hasAudio
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

      // 🔧 FIX: Use failParentOnFailure: false to ensure finalizer runs even when children fail
      await mediaFlowProducer.add({
        name: "finalize-analysis",
        queueName: config.MEDIA_PROCESSING_QUEUE_NAME,
        data: { runId: run.id, mediaId: media.id },
        opts: { 
          jobId: `${run.id}-finalizer`,
          attempts: 3, // Retry finalizer up to 3 times if it fails
          backoff: {
            type: 'fixed',
            delay: 3000, // Wait 3 seconds before retry
          },
          removeOnComplete: {
            age: 86400, // Keep completed finalizer jobs for 24 hours
          },
          removeOnFail: {
            age: 604800, // Keep failed finalizer jobs for 7 days
          },
        },
        children: childJobs.map(job => ({
          ...job,
          opts: {
            ...job.opts,
            failParentOnFailure: false, // 🔧 KEY FIX: Don't fail parent when child fails
          }
        })),
      });
      
      // Emit initial QUEUED events for all models so frontend shows them immediately
      for (const modelName of compatibleModels) {
        try {
          const queuedEvent = {
            media_id: media.id,
            user_id: media.userId,
            event: "ANALYSIS_QUEUED",
            message: `${modelName} queued for analysis`,
            data: {
              model_name: modelName,
              progress: 0,
              total: 100,
              phase: "QUEUED",
              timestamp: new Date().toISOString(),
            },
          };
          await redisPublisher.publish(
            config.MEDIA_PROGRESS_CHANNEL_NAME,
            JSON.stringify(queuedEvent)
          );
          logger.debug(`[MediaService] Emitted QUEUED event for model ${modelName}`);
        } catch (error) {
          logger.error(`[MediaService] Failed to emit QUEUED event for ${modelName}:`, error);
        }
      }
      
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

  async _determineCompatibleModels(mediaType, hasAudio = null) {
    const serverStats = await modelAnalysisService.getServerStatistics();
    if (!serverStats?.models_info) return [];

    return (
      serverStats.models_info
        .filter((m) => {
          if (!m.loaded) return false;
          
          // First check basic media type compatibility
          let isCompatible = false;
          switch (mediaType) {
            case "AUDIO":
              isCompatible = m.isAudio;
              break;
            case "IMAGE":
              isCompatible = m.isImage;
              break;
            case "VIDEO":
              isCompatible = m.isVideo;
              break;
            default:
              return false;
          }
          
          if (!isCompatible) return false;
          
          // For video media, check if model requires audio and if video has audio
          if (mediaType === "VIDEO" && m.isAudio && m.isVideo) {
            // Model requires both video AND audio (like LIP-FD-V1)
            if (hasAudio === false) {
              logger.info(
                `[MediaService] Skipping model ${m.name} - requires audio but video has no audio track`
              );
              return false; // Skip this model
            }
          }
          
          return true;
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
