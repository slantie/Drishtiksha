// src/utils/testModelIntegration.js

import { config } from "dotenv";
import { modelAnalysisService } from "../services/modelAnalysis.service.js";
import logger from "./logger.js";

// Load environment variables first
config();

/**
 * Test script to verify the LSTM model integration
 */
export async function testModelIntegration() {
    logger.info("🧪 Starting LSTM Model Integration Test...");

    try {
        // Test 1: Check if service is configured
        logger.info("Test 1: Checking service configuration...");
        const isAvailable = modelAnalysisService.isAvailable();
        logger.info(`Service configured: ${isAvailable}`);

        if (!isAvailable) {
            logger.warn(
                "❌ Model service is not configured. Please check SERVER_URL and SERVER_API_KEY environment variables."
            );
            return false;
        }

        // Test 2: Health check
        logger.info("Test 2: Performing health check...");
        try {
            const health = await modelAnalysisService.checkHealth();
            logger.info(`✅ Health check passed: ${JSON.stringify(health)}`);
        } catch (error) {
            logger.error(`❌ Health check failed: ${error.message}`);
            return false;
        }

        // Test 3: Get model info
        logger.info("Test 3: Getting model information...");
        try {
            const modelInfo = await modelAnalysisService.getModelInfo();
            logger.info(
                `✅ Model info retrieved: ${JSON.stringify(modelInfo, null, 2)}`
            );
        } catch (error) {
            logger.error(`❌ Model info retrieval failed: ${error.message}`);
            return false;
        }

        // Test 4: Test fallback functionality
        logger.info("Test 4: Testing fallback functionality...");
        try {
            const fallbackResult =
                modelAnalysisService.generateFallbackResult();
            logger.info(
                `✅ Fallback result generated: ${JSON.stringify(
                    fallbackResult,
                    null,
                    2
                )}`
            );
        } catch (error) {
            logger.error(`❌ Fallback test failed: ${error.message}`);
            return false;
        }

        logger.info("🎉 All integration tests passed successfully!");
        return true;
    } catch (error) {
        logger.error(`❌ Integration test failed: ${error.message}`);
        return false;
    }
}

// Run test if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
    testModelIntegration()
        .then((success) => {
            process.exit(success ? 0 : 1);
        })
        .catch((error) => {
            logger.error(`Test execution failed: ${error.message}`);
            process.exit(1);
        });
}
