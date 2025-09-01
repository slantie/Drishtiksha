// server.js

import { createServer } from 'http';
import { app } from './src/app.js';
import { config, connectServices, disconnectServices } from './src/config/index.js';
import logger from './src/utils/logger.js';
import { initializeSocketIO } from './src/config/socket.js';
import { initializeQueueEvents } from './src/workers/queueEvents.js';
import { initializeRedisListener } from './src/services/event.service.js';

const PORT = config.PORT;
const httpServer = createServer(app);

const startServer = async () => {
    try {
        const io = initializeSocketIO(httpServer);
        app.set('io', io);

        initializeQueueEvents(io);
        initializeRedisListener(io);
        
        await connectServices();

        httpServer.listen(PORT, () => {
            logger.info(`🚀 Server is running at: http://localhost:${PORT}`);
            logger.info(`   Environment: ${config.NODE_ENV}`);
        });
        
    } catch (error) {
        logger.error('❌ FATAL: Failed to start server:', error);
        process.exit(1);
    }
};

const shutdown = async (signal) => {
    logger.info(`\n${signal} received. Shutting down gracefully...`);
    httpServer.close(async () => {
        logger.info('HTTP server closed.');
        await disconnectServices();
        logger.info('🔌 All service connections closed. Shutdown complete.');
        process.exit(0);
    });
    setTimeout(() => {
        logger.error('Could not close connections in time, forcefully shutting down.');
        process.exit(1);
    }, 10000);
};

startServer();
process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));