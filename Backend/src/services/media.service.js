// src/services/media.service.js

// RENAMED: Importing mediaRepository
import { mediaRepository } from "../repositories/media.repository.js";
import { addMediaAnalysisFlow } from "../config/queue.js";
import { modelAnalysisService } from "./modelAnalysis.service.js";
import storageManager from "../storage/storage.manager.js";
import { ApiError } from "../utils/ApiError.js";
import logger from "../utils/logger.js";
import { getMediaType } from "../utils/media.js";
import { MEDIA_PROCESSING_QUEUE_NAME } from "../config/constants.js";

// RENAMED: from VideoService to MediaService
class MediaService {
    // RENAMED: from createVideoAndQueueForAnalysis to createMediaAndQueueForAnalysis
    async createMediaAndQueueForAnalysis(file, user, description) {
        if (!file) throw new ApiError(400, "A file is required.");

        // --- NEW: Determine the media type from the file's mimetype ---
        const mediaType = getMediaType(file.mimetype);
        if (!mediaType) {
            throw new ApiError(415, `Unsupported file type: ${file.mimetype}`);
        }

        const uploadResponse = await storageManager.uploadFile(
            file.path,
            mediaType.toLowerCase() + "s"
        );
        if (!uploadResponse) {
            throw new ApiError(500, "Failed to upload file.");
        }

        // --- UPDATED: Create a generic Media record ---
        const newMedia = await mediaRepository.create({
            filename: file.originalname,
            description,
            url: uploadResponse.url,
            publicId: uploadResponse.publicId,
            mimetype: file.mimetype,
            size: file.size,
            status: "QUEUED",
            userId: user.id,
            // --- NEW: Store the determined media type ---
            mediaType: mediaType,
        });

        // --- NEW: Create the specific metadata record ---
        // In a real-world scenario, you would use a library like `fluent-ffmpeg`
        // to extract metadata. For now, we create a placeholder record.
        await mediaRepository.createMetadata(newMedia.id, mediaType, {
            // Placeholder data
            duration: mediaType === "IMAGE" ? null : 0,
            width: mediaType === "AUDIO" ? null : 0,
            height: mediaType === "AUDIO" ? null : 0,
        });

        // --- UPDATED: Analysis queuing is now media-type aware ---
        try {
            logger.info(
                `[MediaService] Creating analysis flow for ${mediaType} ${newMedia.id}`
            );
            const serverStats =
                await modelAnalysisService.getServerStatistics();

            // Filter models based on media type using server-provided flags
            const availableModels =
                serverStats.models_info
                    ?.filter((m) => {
                        // Use server-provided flags to determine model compatibility
                        if (mediaType === "AUDIO") {
                            return m.loaded && m.isAudio === true;
                        }
                        if (mediaType === "VIDEO") {
                            return m.loaded && m.isVideo === true;
                        }
                        if (mediaType === "IMAGE") {
                            // Images can be processed by video models
                            return m.loaded && m.isVideo === true;
                        }
                        return false;
                    })
                    .map((m) => m.name) || [];

            if (availableModels.length === 0) {
                // If no models are available, it's not a failure, but it's fully "ANALYZED" with 0 results.
                await mediaRepository.updateStatus(newMedia.id, "ANALYZED");
                logger.warn(
                    `No compatible models found for ${mediaType} ${newMedia.id}. Marking as analyzed.`
                );
                return newMedia; // Return early
            }

            const childJobs = availableModels.map((modelName) => ({
                name: "run-single-analysis",
                // Pass mediaId and mediaType to the worker
                data: {
                    mediaId: newMedia.id,
                    mediaType,
                    modelName,
                    serverStats,
                },
                queueName: MEDIA_PROCESSING_QUEUE_NAME, // This queue name can remain for now
                opts: { jobId: `${newMedia.id}-${modelName}` },
            }));

            await addMediaAnalysisFlow(newMedia.id, childJobs);

            logger.info(
                `[MediaService] Successfully queued analysis flow for ${mediaType} ${newMedia.id} with ${childJobs.length} models.`
            );
        } catch (error) {
            logger.error(
                `[MediaService] Failed to create analysis flow for media ${newMedia.id}: ${error.message}`
            );
            await mediaRepository.updateStatus(newMedia.id, "FAILED");
            throw new ApiError(
                500,
                "Could not queue media for analysis.",
                error.stack
            );
        }

        return newMedia;
    }

    // RENAMED: from getAllVideosForUser to getAllMediaForUser
    async getAllMediaForUser(userId) {
        return mediaRepository.findAllByUserId(userId);
    }

    // RENAMED: from getVideoWithAnalyses to getMediaWithAnalyses
    async getMediaWithAnalyses(mediaId, userId) {
        const mediaItem = await mediaRepository.findByIdAndUserId(
            mediaId,
            userId
        );
        if (!mediaItem)
            throw new ApiError(
                404,
                "Media not found or you do not have permission."
            );
        return mediaItem;
    }

    // RENAMED: from updateVideo to updateMedia
    async updateMedia(mediaId, userId, updateData) {
        const mediaItem = await this.getMediaWithAnalyses(mediaId, userId); // Ensures user owns the media
        if (!mediaItem) throw new ApiError(404, "Media not found.");

        const allowedFields = ["filename", "description"];
        for (const key of Object.keys(updateData)) {
            if (!allowedFields.includes(key)) {
                throw new ApiError(400, `Field '${key}' is not updatable.`);
            }
        }

        const updatedMedia = await mediaRepository.updateById(
            mediaId,
            updateData
        );
        return updatedMedia;
    }

    // RENAMED: from deleteVideoById to deleteMediaById
    async deleteMediaById(mediaId, userId) {
        const mediaItem = await this.getMediaWithAnalyses(mediaId, userId);

        // --- UPDATED: Use mediaType to inform the storage manager ---
        if (mediaItem.publicId) {
            const resourceType = mediaItem.mediaType.toLowerCase(); // 'video', 'image', or 'audio'
            await storageManager.deleteFile(mediaItem.publicId, resourceType);
        }

        // Note: Logic for deleting associated visualizations can be added here
        // once that feature is fully generic.

        await mediaRepository.deleteById(mediaId);
    }
}

// RENAMED: Exporting mediaService
export const mediaService = new MediaService();
