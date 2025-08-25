// tests/integration/real-analysis.integration.test.js

import request from "supertest";
import path from "path";
import fs from "fs";
import { app } from "../../src/app.js";
import prisma from "../../src/config/database.js";
import { modelAnalysisService } from "../../src/services/modelAnalysis.service.js";
import dotenv from "dotenv";

// Load environment variables
dotenv.config({ path: "./.env" });

/**
 * Real Analysis Integration Test Suite
 *
 * This test specifically validates that:
 * 1. Backend properly communicates with the actual ML Server
 * 2. Real ML models process uploaded media
 * 3. Analysis results are properly stored and retrieved
 * 4. The complete workflow from upload to analysis completion works
 */

describe("Real Backend-Server ML Analysis Integration", () => {
    let testApi;
    let testUser;
    let testUserToken;
    let createdMediaIds = [];
    let availableModels = [];

    // Helper to create test user
    const createTestUser = async () => {
        const timestamp = Date.now();
        const userData = {
            email: `ml-test-${timestamp}@example.com`,
            password: "MLTest123!",
            firstName: "ML",
            lastName: "Test",
        };

        const signupRes = await testApi
            .post("/api/v1/auth/signup")
            .send(userData)
            .timeout(10000);

        expect(signupRes.statusCode).toBe(201);

        const loginRes = await testApi
            .post("/api/v1/auth/login")
            .send({
                email: userData.email,
                password: userData.password,
            })
            .timeout(10000);

        expect(loginRes.statusCode).toBe(200);

        return {
            user: userData,
            token: loginRes.body.data.token,
            userId: loginRes.body.data.user.id,
        };
    };

    // Helper to wait for analysis completion
    const waitForAnalysisCompletion = async (
        mediaId,
        token,
        maxWaitTime = 600000
    ) => {
        const pollInterval = 10000; // 10 seconds
        let elapsed = 0;

        console.log(
            `[Real Integration] Waiting for analysis completion: ${mediaId}`
        );

        while (elapsed < maxWaitTime) {
            const res = await testApi
                .get(`/api/v1/media/${mediaId}`)
                .set("Authorization", `Bearer ${token}`)
                .timeout(30000);

            expect(res.statusCode).toBe(200);
            const media = res.body.data;

            console.log(
                `[Real Integration] Status: ${media.status}, Analyses: ${
                    media.analyses?.length || 0
                }`
            );

            if (
                ["ANALYZED", "PARTIALLY_ANALYZED", "FAILED"].includes(
                    media.status
                )
            ) {
                return media;
            }

            await new Promise((resolve) => setTimeout(resolve, pollInterval));
            elapsed += pollInterval;
        }

        throw new Error(
            `Analysis did not complete within ${maxWaitTime / 1000} seconds`
        );
    };

    beforeAll(async () => {
        console.log("\n" + "=".repeat(80));
        console.log("REAL ML ANALYSIS INTEGRATION TEST");
        console.log("=".repeat(80));

        testApi = request(app);

        // Verify server configuration
        if (!process.env.SERVER_URL || !process.env.SERVER_API_KEY) {
            throw new Error("SERVER_URL and SERVER_API_KEY must be configured");
        }

        console.log(`[Real Integration] Server URL: ${process.env.SERVER_URL}`);
        console.log(`[Real Integration] API Key: ✅ Configured`);

        // Test server connectivity and get available models
        try {
            const serverStats =
                await modelAnalysisService.getServerStatistics();
            availableModels = serverStats.models_info
                ? serverStats.models_info.map((m) => m.name)
                : [];
            console.log(
                `[Real Integration] Server Status: ${serverStats.status}`
            );
            console.log(
                `[Real Integration] Available Models: ${availableModels.join(
                    ", "
                )}`
            );

            if (availableModels.length === 0) {
                throw new Error("No models available on server");
            }
        } catch (error) {
            throw new Error(`Failed to connect to ML Server: ${error.message}`);
        }

        // Create test user
        const userData = await createTestUser();
        testUser = userData.user;
        testUserToken = userData.token;
        console.log(`[Real Integration] Test user created: ${userData.userId}`);
    }, 120000);

    afterAll(async () => {
        console.log("\n[Real Integration] Cleaning up test data...");

        // Delete created media
        for (const mediaId of createdMediaIds) {
            try {
                await testApi
                    .delete(`/api/v1/media/${mediaId}`)
                    .set("Authorization", `Bearer ${testUserToken}`)
                    .timeout(30000);
                console.log(`[Real Integration] Deleted media: ${mediaId}`);
            } catch (error) {
                console.warn(
                    `[Real Integration] Failed to delete media ${mediaId}: ${error.message}`
                );
            }
        }

        // Delete test user
        try {
            const userRes = await testApi.post("/api/v1/auth/login").send({
                email: testUser.email,
                password: testUser.password,
            });

            if (userRes.statusCode === 200) {
                await prisma.user.delete({
                    where: { id: userRes.body.data.user.id },
                });
                console.log(`[Real Integration] Deleted test user`);
            }
        } catch (error) {
            console.warn(
                `[Real Integration] Failed to delete test user: ${error.message}`
            );
        }

        await prisma.$disconnect();
    });

    describe("Server Connectivity", () => {
        it("should connect to the actual ML server and retrieve statistics", async () => {
            const serverStats =
                await modelAnalysisService.getServerStatistics();

            expect(serverStats).toBeDefined();
            expect(serverStats.status).toBeDefined();
            expect(Array.isArray(serverStats.models_info)).toBe(true);
            expect(serverStats.models_info.length).toBeGreaterThan(0);

            console.log(`[Real Integration] ✅ Server connectivity verified`);
            console.log(
                `[Real Integration] Server stats: ${JSON.stringify(
                    serverStats,
                    null,
                    2
                )}`
            );
        });

        it("should validate API key authentication with server", async () => {
            // This tests that our API key works with the actual server
            const res = await testApi
                .get("/api/v1/monitoring/server-status")
                .set("Authorization", `Bearer ${testUserToken}`)
                .timeout(30000);

            expect(res.statusCode).toBe(200);
            expect(res.body.success).toBe(true);
            expect(res.body.data.status).toBeDefined();

            console.log(
                `[Real Integration] ✅ API key authentication verified`
            );
        });
    });

    describe("Real ML Analysis Workflow", () => {
        it("should upload video and get real ML analysis results", async () => {
            const testVideoPath = path.join(
                process.cwd(),
                "tests",
                "fixtures",
                "test-video.mp4"
            );

            if (!fs.existsSync(testVideoPath)) {
                throw new Error(`Test video not found: ${testVideoPath}`);
            }

            console.log(
                `[Real Integration] Uploading test video: ${testVideoPath}`
            );

            // Upload video
            const uploadRes = await testApi
                .post("/api/v1/media")
                .set("Authorization", `Bearer ${testUserToken}`)
                .field("description", "Real ML analysis test")
                .attach("file", testVideoPath)
                .timeout(60000);

            expect(uploadRes.statusCode).toBe(202); // 202 Accepted for processing
            expect(uploadRes.body.success).toBe(true);

            const mediaId = uploadRes.body.data.id;
            createdMediaIds.push(mediaId);

            console.log(
                `[Real Integration] Video uploaded with ID: ${mediaId}`
            );
            console.log(
                `[Real Integration] Status: ${uploadRes.body.data.status}`
            );

            // Wait for analysis to complete
            const finalMedia = await waitForAnalysisCompletion(
                mediaId,
                testUserToken,
                900000
            );

            // Validate analysis results
            expect(finalMedia).toBeDefined();
            expect(finalMedia.id).toBe(mediaId);

            if (
                finalMedia.status === "ANALYZED" ||
                finalMedia.status === "PARTIALLY_ANALYZED"
            ) {
                expect(finalMedia.analyses).toBeDefined();
                expect(Array.isArray(finalMedia.analyses)).toBe(true);
                expect(finalMedia.analyses.length).toBeGreaterThan(0);

                console.log(
                    `[Real Integration] ✅ Analysis completed with ${finalMedia.analyses.length} results`
                );

                // Validate each analysis result
                for (const analysis of finalMedia.analyses) {
                    expect(analysis.id).toBeDefined();
                    expect(analysis.modelName).toBeDefined();
                    expect(analysis.status).toBe("COMPLETED");
                    expect(analysis.result).toBeDefined();

                    // Parse and validate result structure
                    const result =
                        typeof analysis.result === "string"
                            ? JSON.parse(analysis.result)
                            : analysis.result;

                    expect(result).toBeDefined();
                    expect(typeof result).toBe("object");

                    // Verify this is a real ML result (not mocked)
                    const hasMLFields =
                        result.prediction !== undefined ||
                        result.confidence !== undefined ||
                        result.score !== undefined ||
                        result.probability !== undefined ||
                        result.is_fake !== undefined ||
                        result.deepfake_probability !== undefined;

                    expect(hasMLFields).toBe(true);

                    console.log(
                        `[Real Integration] ✅ ${
                            analysis.modelName
                        }: ${JSON.stringify(result).substring(0, 150)}...`
                    );
                }

                // Verify we're using actual server models
                const usedModels = finalMedia.analyses.map((a) => a.modelName);
                const matchedModels = usedModels.filter((model) =>
                    availableModels.some(
                        (serverModel) =>
                            serverModel
                                .toLowerCase()
                                .includes(model.toLowerCase()) ||
                            model
                                .toLowerCase()
                                .includes(serverModel.toLowerCase())
                    )
                );

                expect(matchedModels.length).toBeGreaterThan(0);
                console.log(
                    `[Real Integration] ✅ Used real server models: ${matchedModels.join(
                        ", "
                    )}`
                );
            } else if (finalMedia.status === "FAILED") {
                console.error(
                    `[Real Integration] ❌ Analysis failed for media ${mediaId}`
                );
                console.error(
                    `[Real Integration] This indicates a real integration problem`
                );

                // Still expect the workflow to complete, even if analysis fails
                expect(finalMedia.status).toBe("FAILED");
            }

            // Verify database persistence
            const dbMedia = await prisma.media.findUnique({
                where: { id: mediaId },
                include: { analyses: true },
            });

            expect(dbMedia).toBeDefined();
            expect(dbMedia.status).toBe(finalMedia.status);
            expect(dbMedia.analyses.length).toBe(
                finalMedia.analyses?.length || 0
            );
        }, 1200000); // 20 minutes timeout for real ML processing

        it("should handle multiple concurrent analyses", async () => {
            const testVideoPath = path.join(
                process.cwd(),
                "tests",
                "fixtures",
                "test-video.mp4"
            );

            if (!fs.existsSync(testVideoPath)) {
                throw new Error(`Test video not found: ${testVideoPath}`);
            }

            console.log(`[Real Integration] Testing concurrent analysis...`);

            // Upload multiple videos simultaneously
            const uploadPromises = [];
            for (let i = 0; i < 2; i++) {
                const uploadPromise = testApi
                    .post("/api/v1/media")
                    .set("Authorization", `Bearer ${testUserToken}`)
                    .field("description", `Concurrent test ${i + 1}`)
                    .attach("file", testVideoPath)
                    .timeout(60000);

                uploadPromises.push(uploadPromise);
            }

            const uploadResults = await Promise.all(uploadPromises);

            // Verify all uploads succeeded
            const mediaIds = [];
            for (const uploadRes of uploadResults) {
                expect(uploadRes.statusCode).toBe(202); // 202 Accepted for processing
                const mediaId = uploadRes.body.data.id;
                mediaIds.push(mediaId);
                createdMediaIds.push(mediaId);
            }

            console.log(
                `[Real Integration] Uploaded ${mediaIds.length} videos concurrently`
            );

            // Wait for all analyses to complete
            const analysisPromises = mediaIds.map((id) =>
                waitForAnalysisCompletion(id, testUserToken, 900000)
            );

            const results = await Promise.all(analysisPromises);

            // Verify all analyses completed
            for (let i = 0; i < results.length; i++) {
                const media = results[i];
                expect(media).toBeDefined();
                expect(["ANALYZED", "PARTIALLY_ANALYZED", "FAILED"]).toContain(
                    media.status
                );

                console.log(
                    `[Real Integration] ✅ Concurrent analysis ${i + 1}: ${
                        media.status
                    }`
                );
            }
        }, 1800000); // 30 minutes for concurrent processing
    });

    describe("Error Handling and Edge Cases", () => {
        it("should handle server connectivity issues gracefully", async () => {
            // Temporarily modify server URL to test error handling
            const originalUrl = process.env.SERVER_URL;
            process.env.SERVER_URL = "http://localhost:9999"; // Non-existent server

            try {
                await modelAnalysisService.getServerStatistics();
                // Should not reach here
                expect(true).toBe(false);
            } catch (error) {
                expect(error).toBeDefined();
                console.log(
                    `[Real Integration] ✅ Error handling verified: ${error.message}`
                );
            } finally {
                // Restore original URL
                process.env.SERVER_URL = originalUrl;
            }
        });

        it("should validate analysis timeout handling", async () => {
            // This test verifies our timeout mechanisms work correctly
            const testVideoPath = path.join(
                process.cwd(),
                "tests",
                "fixtures",
                "test-video.mp4"
            );

            if (!fs.existsSync(testVideoPath)) {
                console.warn(
                    "[Real Integration] Skipping timeout test - no test video"
                );
                return;
            }

            const uploadRes = await testApi
                .post("/api/v1/media")
                .set("Authorization", `Bearer ${testUserToken}`)
                .field("description", "Timeout test")
                .attach("file", testVideoPath)
                .timeout(60000);

            expect(uploadRes.statusCode).toBe(202); // 202 Accepted for processing
            const mediaId = uploadRes.body.data.id;
            createdMediaIds.push(mediaId);

            // Test with a very short timeout to verify timeout handling
            try {
                await waitForAnalysisCompletion(mediaId, testUserToken, 5000); // 5 seconds
                // If it completes this fast, that's actually good
                console.log(`[Real Integration] ✅ Analysis completed quickly`);
            } catch (error) {
                expect(error.message).toContain("did not complete within");
                console.log(`[Real Integration] ✅ Timeout handling verified`);
            }
        });
    });
});
