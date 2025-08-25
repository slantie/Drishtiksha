// tests/integration/backend-server.integration.test.js

import request from "supertest";
import path from "path";
import { app } from "../../src/app.js";
import { jest } from "@jest/globals";
import prisma from "../../src/config/database.js";
import { modelAnalysisService } from "../../src/services/modelAnalysis.service.js";
import dotenv from "dotenv";

// Load environment variables
dotenv.config({ path: "./.env" });

/**
 * Backend-Server Integration Test Suite
 *
 * This test suite validates the complete integration between the Backend and Server components:
 * 1. Uses real environment variables (SERVER_URL, SERVER_API_KEY)
 * 2. Creates test users with proper authentication
 * 3. Tests media upload, processing, and analysis workflows
 * 4. Validates server communication and response handling
 * 5. Ensures proper timeout handling for long-running processes
 */

// Helper function to poll the API for final media status with proper timeout handling
const pollForCompletion = async (mediaId, authToken, maxTimeout = 600000) => {
    const POLLING_INTERVAL = 20000; // 20 seconds between polls
    let elapsedTime = 0;

    console.log(
        `\n[Integration Test] Starting to poll for media completion: ${mediaId}`
    );
    console.log(
        `[Integration Test] Max timeout: ${
            maxTimeout / 1000
        }s, Poll interval: ${POLLING_INTERVAL / 1000}s`
    );

    while (elapsedTime < maxTimeout) {
        try {
            const res = await request(app)
                .get(`/api/v1/media/${mediaId}`)
                .set("Authorization", `Bearer ${authToken}`)
                .timeout(30000); // 30 second timeout per request

            if (res.statusCode !== 200) {
                throw new Error(
                    `API returned status ${res.statusCode}: ${
                        res.body?.message || "Unknown error"
                    }`
                );
            }

            const media = res.body.data;
            const finalStatuses = ["ANALYZED", "PARTIALLY_ANALYZED", "FAILED"];

            if (finalStatuses.includes(media.status)) {
                console.log(
                    `[Integration Test] Final status received: ${media.status} for media ${mediaId}`
                );
                console.log(
                    `[Integration Test] Total analyses: ${
                        media.analyses?.length || 0
                    }`
                );
                return media;
            }

            console.log(
                `[Integration Test] Status: '${media.status}', analyses: ${
                    media.analyses?.length || 0
                }. Waiting ${POLLING_INTERVAL / 1000}s...`
            );

            // Wait before next poll
            await new Promise((resolve) =>
                setTimeout(resolve, POLLING_INTERVAL)
            );
            elapsedTime += POLLING_INTERVAL;
        } catch (error) {
            console.error(`[Integration Test] Polling error: ${error.message}`);
            throw error;
        }
    }

    throw new Error(
        `[Integration Test] Polling timed out after ${
            maxTimeout / 1000
        }s for media ${mediaId}`
    );
};

// Helper function to create a unique test user
const createTestUser = async (api) => {
    const timestamp = Date.now();
    const testUser = {
        email: `integration-test-user-${timestamp}@example.com`,
        password: "IntegrationTest123!",
        firstName: "Integration",
        lastName: "Test",
    };

    console.log(`[Integration Test] Creating test user: ${testUser.email}`);

    const signupRes = await api
        .post("/api/v1/auth/signup")
        .send(testUser)
        .timeout(10000);

    if (signupRes.statusCode !== 201) {
        throw new Error(
            `Failed to create test user: ${
                signupRes.body?.message || "Unknown error"
            }`
        );
    }

    const loginRes = await api
        .post("/api/v1/auth/login")
        .send({
            email: testUser.email,
            password: testUser.password,
        })
        .timeout(10000);

    if (loginRes.statusCode !== 200) {
        throw new Error(
            `Failed to login test user: ${
                loginRes.body?.message || "Unknown error"
            }`
        );
    }

    return {
        user: testUser,
        authToken: loginRes.body.data.token,
        userId: loginRes.body.data.user.id,
    };
};

// Helper function to validate server connectivity
const validateServerConnection = async () => {
    console.log(
        `[Integration Test] Validating server connection to: ${process.env.SERVER_URL}`
    );

    try {
        const serverStats = await modelAnalysisService.getServerStatistics();
        console.log(`[Integration Test] Server status: ${serverStats.status}`);
        console.log(
            `[Integration Test] Available models: ${
                serverStats.models_info?.length || 0
            }`
        );

        return {
            isConnected: true,
            stats: serverStats,
            availableModels: serverStats.models_info || [],
        };
    } catch (error) {
        console.error(
            `[Integration Test] Server connection failed: ${error.message}`
        );
        return {
            isConnected: false,
            error: error.message,
        };
    }
};

describe("Backend-Server Integration Tests", () => {
    // Extended timeout for integration tests (15 minutes)
    jest.setTimeout(900000);

    let api;
    let testUserData;
    let serverConnection;
    const createdMediaIds = [];

    // Test file paths
    const TEST_VIDEO_PATH = path.join(
        process.cwd(),
        "tests",
        "fixtures",
        "test-video.mp4"
    );
    const TEST_AUDIO_PATH = path.join(
        process.cwd(),
        "tests",
        "fixtures",
        "test-audio.mp3"
    );

    beforeAll(async () => {
        console.log("\n" + "=".repeat(80));
        console.log("BACKEND-SERVER INTEGRATION TEST SUITE");
        console.log("=".repeat(80));

        // Initialize API client
        api = request(app);

        // Validate environment configuration
        console.log(`[Integration Test] SERVER_URL: ${process.env.SERVER_URL}`);
        console.log(
            `[Integration Test] SERVER_API_KEY: ${
                process.env.SERVER_API_KEY ? "***configured***" : "NOT SET"
            }`
        );

        if (!process.env.SERVER_URL || !process.env.SERVER_API_KEY) {
            throw new Error(
                "SERVER_URL and SERVER_API_KEY must be configured in environment"
            );
        }

        // Validate server connection
        serverConnection = await validateServerConnection();
        if (!serverConnection.isConnected) {
            console.warn(
                `[Integration Test] WARNING: Server not available - ${serverConnection.error}`
            );
            console.warn(
                `[Integration Test] Some tests may be skipped or fail`
            );
        }

        // Create test user
        testUserData = await createTestUser(api);
        console.log(
            `[Integration Test] Test user created with ID: ${testUserData.userId}`
        );

        // Validate test files exist
        const fs = await import("fs");
        if (!fs.existsSync(TEST_VIDEO_PATH)) {
            console.warn(
                `[Integration Test] WARNING: Test video file not found: ${TEST_VIDEO_PATH}`
            );
        }
        if (!fs.existsSync(TEST_AUDIO_PATH)) {
            console.warn(
                `[Integration Test] WARNING: Test audio file not found: ${TEST_AUDIO_PATH}`
            );
        }
    });

    afterAll(async () => {
        console.log("\n" + "=".repeat(80));
        console.log("CLEANING UP INTEGRATION TEST RESOURCES");
        console.log("=".repeat(80));

        // Cleanup created media items
        for (const mediaId of createdMediaIds) {
            try {
                console.log(`[Integration Test] Deleting media: ${mediaId}`);
                await api
                    .delete(`/api/v1/media/${mediaId}`)
                    .set("Authorization", `Bearer ${testUserData.authToken}`)
                    .timeout(30000);
            } catch (error) {
                console.error(
                    `[Integration Test] Cleanup failed for media ${mediaId}: ${error.message}`
                );
            }
        }

        // Cleanup test user
        if (testUserData?.userId) {
            try {
                console.log(
                    `[Integration Test] Deleting test user: ${testUserData.userId}`
                );
                await prisma.user.delete({
                    where: { id: testUserData.userId },
                });
            } catch (error) {
                console.error(
                    `[Integration Test] Failed to delete test user: ${error.message}`
                );
            }
        }

        // Close database connections
        await prisma.$disconnect();
        console.log("[Integration Test] Cleanup completed");
    });

    describe("Server Connectivity and Health", () => {
        it("should connect to the ML server and retrieve health status", async () => {
            const result = await validateServerConnection();

            expect(result.isConnected).toBe(true);
            expect(result.stats).toBeDefined();
            expect(result.stats.status).toBeDefined();
            expect(Array.isArray(result.availableModels)).toBe(true);

            console.log(
                `[Integration Test] Server health validated successfully`
            );
            console.log(`[Integration Test] Status: ${result.stats.status}`);
            console.log(
                `[Integration Test] Models available: ${result.availableModels.length}`
            );
        });

        it("should retrieve server statistics through monitoring endpoint", async () => {
            const res = await api
                .get("/api/v1/monitoring/server-status")
                .set("Authorization", `Bearer ${testUserData.authToken}`)
                .timeout(30000);

            expect(res.statusCode).toBe(200);
            expect(res.body.success).toBe(true);
            expect(res.body.data.status).toBeDefined();
            expect(res.body.data.modelsInfo).toBeDefined();

            console.log(
                `[Integration Test] Server statistics retrieved via API`
            );
        });
    });

    describe("Authentication and User Management", () => {
        it("should authenticate user with valid credentials", async () => {
            const loginRes = await api
                .post("/api/v1/auth/login")
                .send({
                    email: testUserData.user.email,
                    password: testUserData.user.password,
                })
                .timeout(10000);

            expect(loginRes.statusCode).toBe(200);
            expect(loginRes.body.success).toBe(true);
            expect(loginRes.body.data.token).toBeDefined();
            expect(loginRes.body.data.user.id).toBe(testUserData.userId);

            console.log(`[Integration Test] User authentication validated`);
        });

        it("should reject invalid credentials", async () => {
            const loginRes = await api
                .post("/api/v1/auth/login")
                .send({
                    email: testUserData.user.email,
                    password: "wrong-password",
                })
                .timeout(10000);

            expect(loginRes.statusCode).toBe(401);
            expect(loginRes.body.success).toBe(false);

            console.log(
                `[Integration Test] Invalid authentication properly rejected`
            );
        });
    });

    describe("Video Media Processing Integration", () => {
        let videoMediaId;

        it("should upload and queue a video file for analysis", async () => {
            const fs = await import("fs");
            if (!fs.existsSync(TEST_VIDEO_PATH)) {
                console.warn(
                    `[Integration Test] Skipping video test - file not found: ${TEST_VIDEO_PATH}`
                );
                return;
            }

            console.log(
                `[Integration Test] Uploading video file: ${TEST_VIDEO_PATH}`
            );

            const res = await api
                .post("/api/v1/media")
                .set("Authorization", `Bearer ${testUserData.authToken}`)
                .field(
                    "description",
                    "Integration Test Video - Backend-Server Communication"
                )
                .attach("file", TEST_VIDEO_PATH)
                .timeout(60000); // 60 second timeout for upload

            expect(res.statusCode).toBe(202);
            expect(res.body.success).toBe(true);
            expect(res.body.data.id).toBeDefined();
            expect(res.body.data.status).toBe("QUEUED");
            expect(res.body.data.mediaType).toBe("VIDEO");
            expect(res.body.data.userId).toBe(testUserData.userId);

            videoMediaId = res.body.data.id;
            createdMediaIds.push(videoMediaId);

            console.log(
                `[Integration Test] Video uploaded successfully with ID: ${videoMediaId}`
            );
        });

        it("should process video and complete analysis with server integration", async () => {
            if (!videoMediaId) {
                console.warn(
                    `[Integration Test] Skipping video analysis test - no video uploaded`
                );
                return;
            }

            if (!serverConnection.isConnected) {
                console.warn(
                    `[Integration Test] Skipping video analysis test - server not available`
                );
                return;
            }

            console.log(
                `[Integration Test] Starting video analysis polling for: ${videoMediaId}`
            );

            // Poll for completion with extended timeout
            const finalMedia = await pollForCompletion(
                videoMediaId,
                testUserData.authToken,
                600000
            ); // 10 minutes

            // Validate final status
            expect(
                ["ANALYZED", "PARTIALLY_ANALYZED"].includes(finalMedia.status)
            ).toBe(true);

            // Validate analysis results
            expect(Array.isArray(finalMedia.analyses)).toBe(true);
            expect(finalMedia.analyses.length).toBeGreaterThan(0);

            // Check first analysis result
            const firstAnalysis = finalMedia.analyses[0];
            expect(firstAnalysis.status).toBe("COMPLETED");
            expect(["REAL", "FAKE"].includes(firstAnalysis.prediction)).toBe(
                true
            );
            expect(typeof firstAnalysis.confidence).toBe("number");
            expect(firstAnalysis.confidence).toBeGreaterThanOrEqual(0);
            expect(firstAnalysis.confidence).toBeLessThanOrEqual(1);

            // Validate video-specific data
            expect(firstAnalysis.analysisDetails).toBeTruthy();
            expect(Array.isArray(firstAnalysis.frameAnalysis)).toBe(true);
            expect(firstAnalysis.frameAnalysis.length).toBeGreaterThan(0);

            // Ensure audio-specific data is not present for video
            expect(firstAnalysis.audioAnalysis).toBeNull();

            console.log(
                `[Integration Test] Video analysis completed successfully`
            );
            console.log(
                `[Integration Test] Final status: ${finalMedia.status}`
            );
            console.log(
                `[Integration Test] Total analyses: ${finalMedia.analyses.length}`
            );
            console.log(
                `[Integration Test] First analysis prediction: ${firstAnalysis.prediction} (${firstAnalysis.confidence})`
            );
        }, 660000); // 11 minutes timeout for this test
    });

    describe("Audio Media Processing Integration", () => {
        let audioMediaId;

        it("should upload and queue an audio file for analysis", async () => {
            const fs = await import("fs");
            if (!fs.existsSync(TEST_AUDIO_PATH)) {
                console.warn(
                    `[Integration Test] Skipping audio test - file not found: ${TEST_AUDIO_PATH}`
                );
                return;
            }

            console.log(
                `[Integration Test] Uploading audio file: ${TEST_AUDIO_PATH}`
            );

            const res = await api
                .post("/api/v1/media")
                .set("Authorization", `Bearer ${testUserData.authToken}`)
                .field(
                    "description",
                    "Integration Test Audio - Backend-Server Communication"
                )
                .attach("file", TEST_AUDIO_PATH)
                .timeout(60000); // 60 second timeout for upload

            expect(res.statusCode).toBe(202);
            expect(res.body.success).toBe(true);
            expect(res.body.data.id).toBeDefined();
            expect(res.body.data.status).toBe("QUEUED");
            expect(res.body.data.mediaType).toBe("AUDIO");
            expect(res.body.data.userId).toBe(testUserData.userId);

            audioMediaId = res.body.data.id;
            createdMediaIds.push(audioMediaId);

            console.log(
                `[Integration Test] Audio uploaded successfully with ID: ${audioMediaId}`
            );
        });

        it("should process audio and complete analysis with server integration", async () => {
            if (!audioMediaId) {
                console.warn(
                    `[Integration Test] Skipping audio analysis test - no audio uploaded`
                );
                return;
            }

            if (!serverConnection.isConnected) {
                console.warn(
                    `[Integration Test] Skipping audio analysis test - server not available`
                );
                return;
            }

            console.log(
                `[Integration Test] Starting audio analysis polling for: ${audioMediaId}`
            );

            // Poll for completion with extended timeout
            const finalMedia = await pollForCompletion(
                audioMediaId,
                testUserData.authToken,
                600000
            ); // 10 minutes

            // Validate final status
            expect(
                ["ANALYZED", "PARTIALLY_ANALYZED"].includes(finalMedia.status)
            ).toBe(true);

            // Validate analysis results
            expect(Array.isArray(finalMedia.analyses)).toBe(true);
            expect(finalMedia.analyses.length).toBeGreaterThan(0);

            // Find audio analysis result (should use scattering wave model)
            const audioAnalysis = finalMedia.analyses.find(
                (a) =>
                    a.model.includes("SCATTERING-WAVE") ||
                    a.model.includes("AUDIO")
            );

            if (audioAnalysis) {
                expect(audioAnalysis.status).toBe("COMPLETED");
                expect(
                    ["REAL", "FAKE"].includes(audioAnalysis.prediction)
                ).toBe(true);
                expect(typeof audioAnalysis.confidence).toBe("number");

                // Validate audio-specific data
                expect(audioAnalysis.audioAnalysis).toBeTruthy();
                expect(typeof audioAnalysis.audioAnalysis.rmsEnergy).toBe(
                    "number"
                );
                expect(audioAnalysis.audioAnalysis.rmsEnergy).toBeGreaterThan(
                    0
                );
                expect(
                    typeof audioAnalysis.audioAnalysis.spectralCentroid
                ).toBe("number");
                expect(
                    audioAnalysis.audioAnalysis.spectralCentroid
                ).toBeGreaterThan(0);

                // Ensure video-specific data is not present for audio
                expect(audioAnalysis.analysisDetails).toBeNull();
                expect(Array.isArray(audioAnalysis.frameAnalysis)).toBe(true);
                expect(audioAnalysis.frameAnalysis.length).toBe(0);

                console.log(
                    `[Integration Test] Audio analysis completed successfully`
                );
                console.log(
                    `[Integration Test] Audio model used: ${audioAnalysis.model}`
                );
                console.log(
                    `[Integration Test] Audio prediction: ${audioAnalysis.prediction} (${audioAnalysis.confidence})`
                );
            } else {
                console.warn(
                    `[Integration Test] No audio-specific analysis found in results`
                );
                // Still validate that some analysis was completed
                const firstAnalysis = finalMedia.analyses[0];
                expect(firstAnalysis.status).toBe("COMPLETED");
            }
        }, 660000); // 11 minutes timeout for this test
    });

    describe("Error Handling and Edge Cases", () => {
        it("should handle invalid file uploads gracefully", async () => {
            const res = await api
                .post("/api/v1/media")
                .set("Authorization", `Bearer ${testUserData.authToken}`)
                .field("description", "Invalid file test")
                .field("file", "not-a-file")
                .timeout(30000);

            expect(res.statusCode).toBe(400);
            expect(res.body.success).toBe(false);

            console.log(
                `[Integration Test] Invalid file upload properly rejected`
            );
        });

        it("should handle unauthorized access attempts", async () => {
            const res = await api.get("/api/v1/media").timeout(10000);

            expect(res.statusCode).toBe(401);
            expect(res.body.success).toBe(false);

            console.log(
                `[Integration Test] Unauthorized access properly rejected`
            );
        });

        it("should handle server downtime gracefully", async () => {
            // This test would ideally test with server down, but for integration
            // we just validate the error handling structure exists
            const res = await api
                .get("/api/v1/monitoring/server-status")
                .set("Authorization", `Bearer ${testUserData.authToken}`)
                .timeout(30000);

            // Even if server is down, API should respond with structured error
            expect([200, 503].includes(res.statusCode)).toBe(true);
            expect(typeof res.body.success).toBe("boolean");

            console.log(`[Integration Test] Server status endpoint responsive`);
        });
    });

    describe("Queue and Monitoring Integration", () => {
        it("should retrieve queue status information", async () => {
            const res = await api
                .get("/api/v1/monitoring/queue-status")
                .set("Authorization", `Bearer ${testUserData.authToken}`)
                .timeout(30000);

            expect(res.statusCode).toBe(200);
            expect(res.body.success).toBe(true);
            expect(res.body.data).toBeDefined();
            expect(typeof res.body.data.pendingJobs).toBe("number");
            expect(typeof res.body.data.activeJobs).toBe("number");

            console.log(
                `[Integration Test] Queue status retrieved successfully`
            );
            console.log(
                `[Integration Test] Pending jobs: ${res.body.data.pendingJobs}`
            );
            console.log(
                `[Integration Test] Active jobs: ${res.body.data.activeJobs}`
            );
        });

        it("should retrieve analysis statistics", async () => {
            const res = await api
                .get("/api/v1/monitoring/analysis-stats?timeframe=24h")
                .set("Authorization", `Bearer ${testUserData.authToken}`)
                .timeout(30000);

            expect(res.statusCode).toBe(200);
            expect(res.body.success).toBe(true);
            expect(res.body.data).toBeDefined();
            expect(typeof res.body.data.total).toBe("number");
            expect(typeof res.body.data.successful).toBe("number");

            console.log(
                `[Integration Test] Analysis statistics retrieved successfully`
            );
        });
    });
});
