// server.js

import { createServer } from 'http';
import { app } from './src/app.js';
import { config, connectServices, disconnectServices } from './src/config/index.js';
import logger from './src/utils/logger.js';
import { initializeSocketIO } from './src/config/socket.js';
import { initializeQueueEvents } from './src/workers/queueEvents.js';
import { initializeRedisListener } from './src/services/event.service.js';
import { checkAndFinalizeStuckRuns } from './src/scripts/check-stuck-runs.js';
import { verifyAnalysisStatuses, fixIncorrectStatuses } from './src/scripts/verify-analysis-statuses.js';

const PORT = config.PORT;
const httpServer = createServer(app);
let stuckRunsCheckInterval = null;
let statusVerifyInterval = null;

const startServer = async () => {
    try {
        const io = initializeSocketIO(httpServer);
        app.set('io', io);

        initializeQueueEvents(io);
        initializeRedisListener(io);
        
        await connectServices();

        // 🔧 NEW: Start periodic check for stuck analysis runs
        // Check every 30 seconds for runs that might be stuck
        stuckRunsCheckInterval = setInterval(async () => {
            try {
                await checkAndFinalizeStuckRuns();
            } catch (error) {
                logger.error(`[Server] Periodic stuck runs check failed: ${error.message}`);
            }
        }, 30 * 1000); // Every 30 seconds

        // 🔧 NEW: Start periodic status verification and auto-fix
        // Check every 5 minutes for status inconsistencies
        statusVerifyInterval = setInterval(async () => {
            try {
                const result = await verifyAnalysisStatuses();
                if (result.incorrectMedia > 0) {
                    logger.warn(`[Server] Found ${result.incorrectMedia} media items with incorrect statuses. Auto-fixing...`);
                    const fixResult = await fixIncorrectStatuses(result.issues);
                    logger.info(`[Server] Auto-fix complete: ${fixResult.fixed} fixed, ${fixResult.failed} failed`);
                }
            } catch (error) {
                logger.error(`[Server] Periodic status verification failed: ${error.message}`);
            }
        }, 5 * 60 * 1000); // Every 5 minutes

        // Run initial checks on startup
        setTimeout(async () => {
            try {
                logger.info('[Server] Running initial stuck runs check...');
                await checkAndFinalizeStuckRuns();
                
                logger.info('[Server] Running initial status verification...');
                const result = await verifyAnalysisStatuses();
                if (result.incorrectMedia > 0) {
                    logger.warn(`[Server] Found ${result.incorrectMedia} status issues on startup. Fixing...`);
                    await fixIncorrectStatuses(result.issues);
                }
            } catch (error) {
                logger.error(`[Server] Initial checks failed: ${error.message}`);
            }
        }, 10000); // Wait 10 seconds after startup

        httpServer.listen(PORT, '0.0.0.0', () => {
            logger.info(`🚀 Server is running and accessible on your network at port: ${PORT}`);
            logger.info(`   Environment: ${config.NODE_ENV}`);
        });
        
    } catch (error) {
        logger.error('❌ FATAL: Failed to start server:', error);
        process.exit(1);
    }
};

const shutdown = async (signal) => {
    logger.info(`\n${signal} received. Shutting down gracefully...`);
    
    // Stop the periodic checkers
    if (stuckRunsCheckInterval) {
        clearInterval(stuckRunsCheckInterval);
        logger.info('Stopped stuck runs checker.');
    }
    
    if (statusVerifyInterval) {
        clearInterval(statusVerifyInterval);
        logger.info('Stopped status verifier.');
    }
    
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