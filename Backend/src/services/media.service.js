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

      logger.info(
        `[MediaService] Successfully created media record ${
          mediaRecord.id
        } for user ${user.id}${
          hasAudio !== null ? ` (hasAudio: ${hasAudio})` : ""
        }`
      );
      return mediaRecord;
    } catch (error) {
      // If database creation failed but file was uploaded, clean up the file
      if (uploadResponse?.publicId) {
        logger.error(
          `[MediaService] Database creation failed after file upload. Cleaning up file: ${uploadResponse.publicId}`
        );
        try {
          await storageManager.deleteFile(uploadResponse.publicId);
          logger.info(
            `[MediaService] Successfully cleaned up orphaned file: ${uploadResponse.publicId}`
          );
        } catch (deleteError) {
          logger.error(
            `[MediaService] Failed to clean up orphaned file ${uploadResponse.publicId}: ${deleteError.message}`
          );
        }
      }
      throw error;
    }
  }

  async _getMediaMetadata(filePath) {
    return new Promise((resolve, reject) => {
      ffmpeg.ffprobe(filePath, (err, metadata) => {
        if (err) {
          logger.error(`[MediaService] ffprobe failed for ${filePath}:`, err);
          return reject(
            new ApiError(
              422,
              "Could not read media file streams.",
              [],
              err.stack
            )
          );
        }
        const hasVideoStream = metadata.streams.some(
          (s) => s.codec_type === "video"
        );
        const hasAudioStream = metadata.streams.some(
          (s) => s.codec_type === "audio"
        );
        resolve({
          hasVideo: hasVideoStream,
          hasAudio: hasAudioStream,
          metadata: metadata.format,
        });
      });
    });
  }

  async _queueAnalysisRun(media, isRerun = false) {
    const latestRunNumber = isRerun
      ? await mediaRepository.findLatestRunNumber(media.id)
      : 0;
    const nextRunNumber = latestRunNumber + 1;

    try {
      logger.info(
        `[MediaService] Preparing analysis run #${nextRunNumber} for media ${media.id}`
      );

      const compatibleModels = await this._determineCompatibleModels(
        media.mediaType,
        media.hasAudio
      );

      if (compatibleModels.length === 0) {
        logger.warn(
          `No compatible models found for ${media.mediaType}${
            media.hasAudio !== null ? ` (hasAudio: ${media.hasAudio})` : ""
          }. Marking as ANALYZED.`
        );
        const run = await mediaRepository.createAnalysisRun(
          media.id,
          nextRunNumber
        );
        await mediaRepository.updateRunStatus(run.id, "ANALYZED");
        await mediaRepository.update(media.id, {
          status: "ANALYZED",
          latestAnalysisRunId: run.id,
        });
        return;
      }

      const run = await mediaRepository.createAnalysisRun(
        media.id,
        nextRunNumber
      );
      await mediaRepository.update(media.id, { status: "QUEUED" });

      const childJobs = compatibleModels.map((modelName) => ({
        name: "run-single-analysis",
        queueName: config.MEDIA_PROCESSING_QUEUE_NAME,
        data: { mediaId: media.id, runId: run.id, modelName },
        opts: {
          jobId: `${run.id}-${modelName}`,
          attempts: 3,
          backoff: {
            type: "exponential",
            delay: 5000,
          },
          removeOnComplete: {
            age: 86400,
            count: 1000,
          },
          removeOnFail: {
            age: 604800,
          },
          failParentOnFailure: false,
        },
      }));

      await mediaFlowProducer.add({
        name: "finalize-analysis",
        queueName: config.MEDIA_PROCESSING_QUEUE_NAME,
        data: { runId: run.id, mediaId: media.id },
        opts: {
          jobId: `${run.id}-finalizer`,
          attempts: 3,
          backoff: {
            type: "fixed",
            delay: 3000,
          },
          removeOnComplete: {
            age: 86400,
          },
          removeOnFail: {
            age: 604800,
          },
        },
        children: childJobs,
      });

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
          logger.debug(
            `[MediaService] Emitted QUEUED event for model ${modelName}`
          );
        } catch (error) {
          logger.error(
            `[MediaService] Failed to emit QUEUED event for ${modelName}:`,
            error
          );
        }
      }

      logger.info(
        `[MediaService] Successfully queued run #${nextRunNumber} for media ${media.id} with ${childJobs.length} models.`
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

          // Handle multimodal models first
          if (m.isMultiModal) {
            // A multimodal model requires both video and audio.
            // It only makes sense for media of type VIDEO.
            const isCompatible = mediaType === "VIDEO" && hasAudio === true;
            if (!isCompatible) {
              logger.info(
                `[MediaService] Skipping multimodal model ${m.name}: requires video with audio.`
              );
            }
            return isCompatible;
          }

          // Handle single-modal models
          switch (mediaType) {
            case "VIDEO":
              return m.isVideo;
            case "AUDIO":
              return m.isAudio;
            case "IMAGE":
              return m.isImage;
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
