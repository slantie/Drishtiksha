// src/workers/queueEvents.js

import { QueueEvents } from 'bullmq';
import { config, prisma, redisConnectionOptionsForBullMQ } from '../config/index.js';
import { mediaRepository } from '../repositories/media.repository.js';
import logger from '../utils/logger.js';

export function initializeQueueEvents(io) {
    const queueEvents = new QueueEvents(config.MEDIA_PROCESSING_QUEUE_NAME, {
        connection: redisConnectionOptionsForBullMQ,
    });

    queueEvents.on('completed', async ({ jobId, returnvalue }) => {
        if (jobId.endsWith('-finalizer')) {
            const { mediaId } = returnvalue;
            logger.info(`[QueueEvents] Finalizer job for media ${mediaId} completed.`);
            
            const media = await mediaRepository.findById(mediaId);
            if (media && media.user) {
                io.to(media.user.id).emit('media_update', media);
                logger.info(`[SocketIO] Emitted final 'media_update' for media ${mediaId} to user ${media.user.id}`);
            }
        }
    });

    queueEvents.on('failed', async ({ jobId, failedReason }) => {
        logger.error(`[QueueEvents] Job ${jobId} failed: ${failedReason}`);
        const runId = jobId.split('-')[0];

        try {
            const run = await prisma.analysisRun.findUnique({
                where: { id: runId },
                include: { media: { select: { userId: true, id: true } } },
            });
            
            if (run?.media) {
                io.to(run.media.userId).emit('processing_error', {
                    mediaId: run.media.id,
                    runId,
                    jobId,
                    error: failedReason,
                });
                logger.info(`[SocketIO] Emitted 'processing_error' for job ${jobId} to user ${run.media.userId}.`);
            }
        } catch (error) {
            logger.error(`[QueueEvents] Error handling failed job notification for ${jobId}: ${error.message}`);
        }
    });

    logger.info('🎧 BullMQ QueueEvents listener initialized.');
    return queueEvents;
}