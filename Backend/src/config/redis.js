// src/config/redis.js

import Redis from 'ioredis';
import { config } from './env.js';
import logger from '../utils/logger.js';

const connectionOptions = {
    host: new URL(config.REDIS_URL).hostname,
    port: parseInt(new URL(config.REDIS_URL).port),
    maxRetriesPerRequest: null,
    lazyConnect: true,
};

// This separate options object is for BullMQ, which manages its own connections.
export const redisConnectionOptionsForBullMQ = {
    ...connectionOptions,
    lazyConnect: undefined, // Let BullMQ control its connection.
};

// Create the singleton clients for Pub/Sub.
export const redisSubscriber = new Redis(connectionOptions);
export const redisPublisher = new Redis(connectionOptions);

/**
 * An idempotent function to ensure a Redis client is connected.
 * It uses a promise attached to the client instance itself to handle
 * concurrent connection attempts gracefully and avoid race conditions.
 * @param {Redis} client - The ioredis client instance.
 * @param {string} clientName - A name for logging (e.g., 'Subscriber').
 * @returns {Promise<void>}
 */
const ensureConnected = (client, clientName) => {
    if (client.connectionPromise) return client.connectionPromise;
    if (client.status === 'ready') return Promise.resolve();

    client.connectionPromise = new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
            reject(new Error(`Connection timeout for Redis client '${clientName}'`));
        }, 5000);

        client.once('ready', () => {
            logger.info(`💌 Redis client '${clientName}' is ready.`);
            clearTimeout(timeout);
            resolve();
        });

        client.once('error', (err) => {
            logger.error(`❌ Redis client '${clientName}' connection error:`, err);
            clearTimeout(timeout);
            reject(err);
        });

        client.connect().catch(err => {
            clearTimeout(timeout);
            reject(err);
        });
    });

    return client.connectionPromise;
};

export const connectRedisClients = async () => {
    await Promise.all([
        ensureConnected(redisSubscriber, 'Subscriber'),
        ensureConnected(redisPublisher, 'Publisher'),
    ]);
};

export const disconnectRedisClients = () => {
    if (redisSubscriber.status === 'ready') redisSubscriber.disconnect();
    if (redisPublisher.status === 'ready') redisPublisher.disconnect();
};