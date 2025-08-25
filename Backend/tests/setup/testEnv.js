// tests/setup/testEnv.js

import dotenv from "dotenv";
import { jest } from "@jest/globals";

// Load environment variables from .env file
dotenv.config({ path: "./.env" });

/**
 * Test Environment Configuration
 *
 * This file sets up the test environment with proper configuration for:
 * 1. Server API integration using real API key
 * 2. Database connections for test isolation
 * 3. Extended timeouts for integration tests
 * 4. Proper cleanup mechanisms
 */

// Validate required environment variables for integration tests
const requiredEnvVars = [
    "SERVER_URL",
    "SERVER_API_KEY",
    "DATABASE_URL",
    "JWT_SECRET",
    "JWT_REFRESH_SECRET",
];

const missingEnvVars = requiredEnvVars.filter((envVar) => !process.env[envVar]);

if (missingEnvVars.length > 0) {
    console.error("❌ Missing required environment variables for tests:");
    missingEnvVars.forEach((envVar) => {
        console.error(`   - ${envVar}`);
    });
    console.error("\nPlease ensure your .env file is properly configured.");
    process.exit(1);
}

// Override test-specific environment variables
process.env.NODE_ENV = "test";

// Use local storage for tests to avoid Cloudinary costs
process.env.STORAGE_PROVIDER = "local";

// Use a test-specific Redis namespace if running tests
if (process.env.REDIS_URL && !process.env.REDIS_URL.includes("test")) {
    console.log(
        "📝 Note: Using production Redis instance for tests. Consider using a test database."
    );
}

// Extended timeouts for integration tests
const isIntegrationTest = process.argv.some(
    (arg) =>
        arg.includes("integration") ||
        arg.includes("e2e") ||
        process.env.TEST_TYPE === "integration"
);

if (isIntegrationTest) {
    // Set longer default timeout for integration tests
    jest.setTimeout(900000); // 15 minutes
    console.log("⏱️  Extended timeout set for integration tests (15 minutes)");
}

// Test database configuration warning
if (
    process.env.DATABASE_URL.includes("localhost") ||
    process.env.DATABASE_URL.includes("127.0.0.1")
) {
    console.log("🏠 Using local database for tests");
} else {
    console.warn("⚠️  WARNING: Tests are running against a remote database!");
    console.warn("   This may interfere with production data.");
    console.warn("   Consider using a local test database.");
}

// Server connection validation
console.log(`🔗 Server URL: ${process.env.SERVER_URL}`);
console.log(
    `🔑 API Key configured: ${process.env.SERVER_API_KEY ? "✅" : "❌"}`
);

// Export test configuration
export const TEST_CONFIG = {
    // Server configuration
    serverUrl: process.env.SERVER_URL,
    serverApiKey: process.env.SERVER_API_KEY,

    // Test timeouts
    defaultTimeout: isIntegrationTest ? 900000 : 30000,
    pollingInterval: 20000, // 20 seconds between status polls
    maxPollingTimeout: 600000, // 10 minutes max for analysis completion

    // Test file paths
    testVideoPath: "./tests/fixtures/test-video.mp4",
    testAudioPath: "./tests/fixtures/test-audio.mp3",

    // Database
    databaseUrl: process.env.DATABASE_URL,

    // Authentication
    jwtSecret: process.env.JWT_SECRET,

    // Storage
    storageProvider: process.env.STORAGE_PROVIDER,

    // Feature flags for tests
    skipServerTests: !process.env.SERVER_URL || !process.env.SERVER_API_KEY,
    skipFileUploadTests: false,

    // Cleanup settings
    cleanupAfterTests: true,
    preserveTestData: process.env.PRESERVE_TEST_DATA === "true",
};

// Global test utilities
export const testUtils = {
    /**
     * Generate a unique test identifier
     */
    generateTestId: () =>
        `test-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,

    /**
     * Create test user data
     */
    createTestUserData: (suffix = null) => {
        const id = suffix || testUtils.generateTestId();
        return {
            email: `test-user-${id}@example.com`,
            password: "TestPassword123!",
            firstName: "Test",
            lastName: "User",
        };
    },

    /**
     * Wait for a specified duration
     */
    wait: (ms) => new Promise((resolve) => setTimeout(resolve, ms)),

    /**
     * Retry a function with exponential backoff
     */
    retry: async (fn, maxAttempts = 3, initialDelay = 1000) => {
        let lastError;

        for (let attempt = 1; attempt <= maxAttempts; attempt++) {
            try {
                return await fn();
            } catch (error) {
                lastError = error;

                if (attempt === maxAttempts) {
                    throw error;
                }

                const delay = initialDelay * Math.pow(2, attempt - 1);
                console.log(
                    `Attempt ${attempt} failed, retrying in ${delay}ms: ${error.message}`
                );
                await testUtils.wait(delay);
            }
        }

        throw lastError;
    },
};

// Setup global error handling for tests
process.on("unhandledRejection", (reason, promise) => {
    console.error("Unhandled Rejection at:", promise, "reason:", reason);
});

process.on("uncaughtException", (error) => {
    console.error("Uncaught Exception:", error);
    process.exit(1);
});

// Log test environment setup completion
console.log("✅ Test environment configuration loaded successfully");

export default TEST_CONFIG;
