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
        
        // Handle finalizer job failures separately
        if (jobId.endsWith('-finalizer')) {
            const mediaId = jobId.replace('-finalizer', '');
            logger.error(`[QueueEvents] Finalizer job failed for media ${mediaId}. Attempting to mark media as FAILED.`);
            
            try {
                const media = await mediaRepository.findById(mediaId);
                if (media) {
                    await mediaRepository.update(mediaId, { status: 'FAILED' });
                    
                    if (media.user) {
                        io.to(media.user.id).emit('processing_error', {
                            mediaId: media.id,
                            filename: media.filename,
                            error: `Finalization failed: ${failedReason}`,
                        });
                    }
                    logger.info(`[QueueEvents] Marked media ${mediaId} as FAILED after finalizer failure.`);
                }
            } catch (error) {
                logger.error(`[QueueEvents] Failed to handle finalizer job failure for ${mediaId}: ${error.message}`);
            }
            return;
        }

        // Handle regular analysis job failures
        const runId = jobId.split('-')[0];

        try {
            const run = await prisma.analysisRun.findUnique({
                where: { id: runId },
                include: { media: { select: { userId: true, id: true, filename: true } } },
            });
            
            if (run?.media) {
                // Emit error notification to user
                io.to(run.media.userId).emit('processing_error', {
                    mediaId: run.media.id,
                    filename: run.media.filename,
                    runId,
                    jobId,
                    error: failedReason,
                });
                logger.info(`[SocketIO] Emitted 'processing_error' for job ${jobId} to user ${run.media.userId}.`);
                
                // Check if all child jobs for this run have failed
                // If so, update the run and media status
                const allAnalyses = await prisma.deepfakeAnalysis.findMany({
                    where: { analysisRunId: runId },
                    select: { status: true },
                });
                
                const allFailed = allAnalyses.length > 0 && allAnalyses.every(a => a.status === 'FAILED');
                
                if (allFailed) {
                    logger.warn(`[QueueEvents] All analyses failed for run ${runId}. Marking as FAILED.`);
                    await mediaRepository.updateRunStatus(runId, 'FAILED');
                    await mediaRepository.update(run.media.id, { 
                        status: 'FAILED',
                        latestAnalysisRunId: runId 
                    });
                }
            }
        } catch (error) {
            logger.error(`[QueueEvents] Error handling failed job notification for ${jobId}: ${error.message}`);
        }
    });

    logger.info('🎧 BullMQ QueueEvents listener initialized.');
    return queueEvents;
}