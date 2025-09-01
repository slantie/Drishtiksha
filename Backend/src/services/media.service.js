// src/services/media.service.js

import { mediaRepository } from '../repositories/media.repository.js';
import { config, mediaFlowProducer } from '../config/index.js';
import { modelAnalysisService } from './modelAnalysis.service.js';
import storageManager from '../storage/storage.manager.js';
import { ApiError } from '../utils/ApiError.js';
import logger from '../utils/logger.js';
import { getMediaType } from '../utils/media.js';

class MediaService {
    async createAndAnalyzeMedia(file, user, description) {
        if (!file) throw new ApiError(400, 'A file is required for upload.');
        const mediaRecord = await this._createMediaRecord(file, user, description);
        await this._queueAnalysisRun(mediaRecord, 1);
        return mediaRecord;
    }

    async rerunAnalysis(mediaId, userId) {
        const media = await mediaRepository.findByIdAndUserId(mediaId, userId);
        if (!media) throw new ApiError(404, 'Media not found or you do not have permission.');
        const latestRunNumber = await mediaRepository.findLatestRunNumber(mediaId);
        const nextRunNumber = latestRunNumber + 1;
        await this._queueAnalysisRun(media, nextRunNumber);
        return await mediaRepository.findById(mediaId);
    }

    async _createMediaRecord(file, user, description) {
        const mediaType = getMediaType(file.mimetype);
        if (!mediaType || mediaType === 'UNKNOWN') {
            throw new ApiError(415, `Unsupported file type: ${file.mimetype}`);
        }
        const uploadResponse = await storageManager.uploadFile(file.path, mediaType.toLowerCase() + 's');
        if (!uploadResponse) throw new ApiError(500, 'Failed to upload file to storage.');
        return mediaRepository.create({
            filename: file.originalname, description, url: uploadResponse.url,
            publicId: uploadResponse.publicId, mimetype: file.mimetype, size: file.size,
            status: 'QUEUED', userId: user.id, mediaType: mediaType,
        });
    }

    async _queueAnalysisRun(media, runNumber) {
        try {
            logger.info(`[MediaService] Preparing analysis run #${runNumber} for media ${media.id}`);
            const compatibleModels = await this._determineCompatibleModels(media.mediaType);
            const run = await mediaRepository.createAnalysisRun(media.id, runNumber);
            await mediaRepository.update(media.id, { status: 'QUEUED' });

            if (compatibleModels.length === 0) {
                logger.warn(`No compatible models found for ${media.mediaType}. Marking run ${run.id} as analyzed.`);
                await mediaRepository.updateRunStatus(run.id, 'ANALYZED');
                await mediaRepository.update(media.id, { status: 'ANALYZED', latestAnalysisRunId: run.id });
                return;
            }

            const childJobs = compatibleModels.map((modelName) => ({
                name: 'run-single-analysis',
                queueName: config.MEDIA_PROCESSING_QUEUE_NAME,
                data: { mediaId: media.id, runId: run.id, modelName },
                opts: { jobId: `${run.id}-${modelName}` },
            }));

            await mediaFlowProducer.add({
                name: 'finalize-analysis',
                queueName: config.MEDIA_PROCESSING_QUEUE_NAME,
                data: { runId: run.id, mediaId: media.id },
                opts: { jobId: `${run.id}-finalizer` },
                children: childJobs,
            });
            logger.info(`[MediaService] Successfully queued run #${runNumber} for media ${media.id} with ${childJobs.length} models.`);
        } catch (error) {
            logger.error(`[MediaService] Failed to create analysis flow for media ${media.id}:`, error);
            await mediaRepository.update(media.id, { status: 'FAILED' });
            throw new ApiError(500, 'Could not queue media for analysis.', [], error.stack);
        }
    }

    async _determineCompatibleModels(mediaType) {
        const serverStats = await modelAnalysisService.getServerStatistics();
        return serverStats.models_info
            ?.filter(m => m.loaded && ((mediaType === 'AUDIO' && m.isAudio) || (mediaType !== 'AUDIO' && m.isVideo)))
            .map(m => m.name) || [];
    }

    async getAllMediaForUser(userId) {
        return mediaRepository.findAllByUserId(userId);
    }

    async getMediaWithAnalyses(mediaId, userId) {
        const mediaItem = await mediaRepository.findByIdAndUserId(mediaId, userId);
        if (!mediaItem) throw new ApiError(404, 'Media not found or you do not have permission.');
        return mediaItem;
    }

    async updateMedia(mediaId, userId, updateData) {
        await this.getMediaWithAnalyses(mediaId, userId); // Ensures ownership
        const allowedFields = ['description'];
        Object.keys(updateData).forEach(key => {
            if (!allowedFields.includes(key)) throw new ApiError(400, `Field '${key}' cannot be updated.`);
        });
        return mediaRepository.update(mediaId, updateData);
    }

    async deleteMediaById(mediaId, userId) {
        const mediaItem = await this.getMediaWithAnalyses(mediaId, userId);
        if (mediaItem.publicId) {
            const resourceType = mediaItem.mediaType.toLowerCase();
            await storageManager.deleteFile(mediaItem.publicId, resourceType);
        }
        await mediaRepository.deleteById(mediaId);
    }
}

export const mediaService = new MediaService();