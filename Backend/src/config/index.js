// src/config/index.js

import { PrismaClient } from '@prisma/client';
import { Queue, FlowProducer } from 'bullmq';
import { config } from './env.js';
import { connectRedisClients, disconnectRedisClients, redisConnectionOptionsForBullMQ } from './redis.js';
import logger from '../utils/logger.js';

export {config} from './env.js';
export {redisConnectionOptionsForBullMQ} from './redis.js';

// --- SERVICE SINGLETON INSTANTIATION ---
export const prisma = new PrismaClient({
    log: config.NODE_ENV === 'development' ? ['warn', 'error'] : ['error'],
});

export const mediaQueue = new Queue(config.MEDIA_PROCESSING_QUEUE_NAME, {
    connection: redisConnectionOptionsForBullMQ,
    defaultJobOptions: {
        attempts: 3,
        backoff: { type: 'exponential', delay: 10000 },
    },
});

export const mediaFlowProducer = new FlowProducer({
    connection: redisConnectionOptionsForBullMQ,
});


// --- CENTRALIZED CONNECTION MANAGEMENT ---
export const connectServices = async () => {
    try {
        await prisma.$connect();
        logger.info('🗄️  Database connected successfully.');
        await connectRedisClients();
    } catch (error) {
        logger.error('❌ Failed to connect to external services:', error);
        process.exit(1);
    }
};

export const disconnectServices = async () => {
    await mediaQueue.close();
    await mediaFlowProducer.close();
    await prisma.$disconnect();
    disconnectRedisClients();
    logger.info('🔌 All external service connections closed.');
};