#!/usr/bin/env node

/**
 * Integration Test Runner
 *
 * This script runs integration tests with proper environment setup and validation.
 * It ensures all required services are available before running tests.
 */

import { spawn } from "child_process";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import fs from "fs";
import axios from "axios";
import dotenv from "dotenv";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const projectRoot = join(__dirname, "..");

// Load environment variables
dotenv.config({ path: join(projectRoot, ".env") });

console.log("🧪 Integration Test Runner Starting...");
console.log("=====================================\n");

// Configuration
const config = {
    serverUrl: process.env.SERVER_URL || "http://localhost:8000",
    serverApiKey: process.env.SERVER_API_KEY,
    databaseUrl: process.env.DATABASE_URL,
    redisUrl: process.env.REDIS_URL,
    testTimeout: 900000, // 15 minutes
};

// Validation functions
const checkServerConnection = async () => {
    console.log(`🔗 Checking server connection: ${config.serverUrl}`);

    if (!config.serverApiKey) {
        console.warn("⚠️  WARNING: SERVER_API_KEY not configured");
        return false;
    }

    try {
        const response = await axios.get(`${config.serverUrl}/stats`, {
            headers: {
                "X-API-Key": config.serverApiKey,
            },
            timeout: 10000,
        });

        console.log(`✅ Server is accessible (Status: ${response.status})`);
        console.log(
            `   Models available: ${response.data.models_info?.length || 0}`
        );
        return true;
    } catch (error) {
        console.error(`❌ Server connection failed: ${error.message}`);
        if (error.code === "ECONNREFUSED") {
            console.error(
                "   Make sure the Python ML server is running on port 8000"
            );
        }
        return false;
    }
};

const checkDatabaseConnection = async () => {
    console.log("🗄️  Checking database connection...");

    if (!config.databaseUrl) {
        console.error("❌ DATABASE_URL not configured");
        return false;
    }

    try {
        // Import prisma dynamically to avoid module loading issues
        const { PrismaClient } = await import("@prisma/client");
        const prisma = new PrismaClient();

        await prisma.$connect();
        console.log("✅ Database connection successful");

        await prisma.$disconnect();
        return true;
    } catch (error) {
        console.error(`❌ Database connection failed: ${error.message}`);
        return false;
    }
};

const checkTestFixtures = () => {
    console.log("📁 Checking test fixtures...");

    const fixtures = [
        join(projectRoot, "tests/fixtures/test-video.mp4"),
        join(projectRoot, "tests/fixtures/test-audio.mp3"),
    ];

    let allFound = true;

    fixtures.forEach((fixture) => {
        if (fs.existsSync(fixture)) {
            console.log(`✅ Found: ${fixture}`);
        } else {
            console.warn(`⚠️  Missing: ${fixture}`);
            allFound = false;
        }
    });

    return allFound;
};

const runTests = async (testPattern = null) => {
    console.log("\n🚀 Running integration tests...");
    console.log("===============================\n");

    const jestArgs = [
        "--verbose",
        "--detectOpenHandles",
        "--forceExit",
        `--testTimeout=${config.testTimeout}`,
        "--runInBand", // Run tests serially for integration tests
    ];

    if (testPattern) {
        jestArgs.push(`--testPathPattern=${testPattern}`);
    } else {
        jestArgs.push("--testPathPattern=integration");
    }

    return new Promise((resolve, reject) => {
        const jest = spawn(
            "node",
            [
                "--experimental-vm-modules",
                "node_modules/jest/bin/jest.js",
                ...jestArgs,
            ],
            {
                cwd: projectRoot,
                stdio: "inherit",
                env: {
                    ...process.env,
                    NODE_ENV: "test",
                    TEST_TYPE: "integration",
                },
            }
        );

        jest.on("close", (code) => {
            if (code === 0) {
                console.log("\n✅ All integration tests passed!");
                resolve();
            } else {
                console.error(`\n❌ Tests failed with exit code ${code}`);
                reject(new Error(`Tests failed with exit code ${code}`));
            }
        });

        jest.on("error", (error) => {
            console.error("❌ Failed to start test runner:", error);
            reject(error);
        });
    });
};

// Main execution
const main = async () => {
    try {
        console.log("Environment Configuration:");
        console.log(`  SERVER_URL: ${config.serverUrl}`);
        console.log(
            `  API Key: ${config.serverApiKey ? "✅ Configured" : "❌ Missing"}`
        );
        console.log(
            `  Database: ${config.databaseUrl ? "✅ Configured" : "❌ Missing"}`
        );
        console.log(
            `  Redis: ${config.redisUrl ? "✅ Configured" : "❌ Missing"}`
        );
        console.log("");

        // Pre-flight checks
        const checks = await Promise.all([
            checkServerConnection(),
            checkDatabaseConnection(),
            Promise.resolve(checkTestFixtures()),
        ]);

        const [serverOk, dbOk, fixturesOk] = checks;

        if (!dbOk) {
            console.error(
                "\n❌ Critical: Database connection required for integration tests"
            );
            process.exit(1);
        }

        if (!serverOk) {
            console.warn(
                "\n⚠️  Warning: Server not available - some tests may fail or be skipped"
            );
            console.warn(
                "   To start the server: cd ../Server && python main.py"
            );
        }

        if (!fixturesOk) {
            console.warn(
                "\n⚠️  Warning: Some test fixtures missing - related tests may be skipped"
            );
        }

        console.log("\n📋 Pre-flight check summary:");
        console.log(`   Database: ${dbOk ? "✅" : "❌"}`);
        console.log(`   Server: ${serverOk ? "✅" : "⚠️"}`);
        console.log(`   Fixtures: ${fixturesOk ? "✅" : "⚠️"}`);

        // Run tests
        const testPattern = process.argv[2];
        await runTests(testPattern);
    } catch (error) {
        console.error("\n❌ Integration test runner failed:", error.message);
        process.exit(1);
    }
};

// Handle process signals
process.on("SIGINT", () => {
    console.log("\n\n🛑 Test runner interrupted by user");
    process.exit(130);
});

process.on("SIGTERM", () => {
    console.log("\n\n🛑 Test runner terminated");
    process.exit(143);
});

// Run if this file is executed directly
if (process.argv[1] === __filename) {
    main().catch((error) => {
        console.error("Fatal error:", error);
        process.exit(1);
    });
}
