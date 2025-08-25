// tests/media.e2e.test.js

import request from "supertest";
import path from "path";
import { app } from "../src/app.js";
import { jest } from "@jest/globals";
import prisma from "../src/config/database.js";

// Helper function to poll the API for a final media status
const pollForCompletion = async (mediaId, authToken) => {
    const POLLING_INTERVAL = 15000; // 15 seconds
    const TIMEOUT = 300000; // 5 minutes
    let elapsedTime = 0;

    console.log(`\n[Polling] Starting to poll for media item: ${mediaId}`);

    while (elapsedTime < TIMEOUT) {
        const res = await request(app)
            .get(`/api/v1/media/${mediaId}`)
            .set("Authorization", `Bearer ${authToken}`);

        const media = res.body.data;
        if (
            ["ANALYZED", "PARTIALLY_ANALYZED", "FAILED"].includes(media.status)
        ) {
            console.log(
                `[Polling] Final status received: ${media.status} for media ${mediaId}`
            );
            return media;
        }

        console.log(
            `[Polling] Status is '${media.status}'. Waiting ${
                POLLING_INTERVAL / 1000
            }s...`
        );
        await new Promise((resolve) => setTimeout(resolve, POLLING_INTERVAL));
        elapsedTime += POLLING_INTERVAL;
    }

    throw new Error(
        `Polling timed out after ${TIMEOUT / 1000}s for media ${mediaId}`
    );
};

describe("Generic Media Analysis E2E Workflow", () => {
    // Set a long timeout for the entire test suite
    jest.setTimeout(420000); // 7 minutes to be safe

    let api;
    let authToken;
    let userId;
    const createdMediaIds = []; // Keep track of created media for cleanup

    // Test media file paths
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

    // 1. Create and Login User before all tests
    beforeAll(async () => {
        api = request(app);
        const testUser = {
            email: `e2e-media-user-${Date.now()}@example.com`,
            password: "password123",
            firstName: "E2E",
            lastName: "Tester",
        };

        // Create user
        await api.post("/api/v1/auth/signup").send(testUser);

        // Login to get token and user ID
        const loginRes = await api.post("/api/v1/auth/login").send({
            email: testUser.email,
            password: testUser.password,
        });

        authToken = loginRes.body.data.token;
        userId = loginRes.body.data.user.id;

        expect(authToken).toBeDefined();
        expect(userId).toBeDefined();
    });

    // 7. Cleanup after all tests are done
    afterAll(async () => {
        // Delete all created media items via the API
        for (const mediaId of createdMediaIds) {
            try {
                await api
                    .delete(`/api/v1/media/${mediaId}`)
                    .set("Authorization", `Bearer ${authToken}`);
            } catch (error) {
                console.error(
                    `Cleanup failed for media ${mediaId}:`,
                    error.message
                );
            }
        }

        // Delete the user directly from the database to ensure a clean state
        if (userId) {
            await prisma.user.delete({ where: { id: userId } });
        }
    });

    // --- VIDEO ANALYSIS TEST ---
    describe("Video Media Type", () => {
        let videoMediaId;

        it("3. should upload a VIDEO file and queue it for analysis", async () => {
            const res = await api
                .post("/api/v1/media") // Use the new generic endpoint
                .set("Authorization", `Bearer ${authToken}`)
                .field("description", "E2E Video Test")
                .attach("file", TEST_VIDEO_PATH); // Use the new generic field name 'file'

            expect(res.statusCode).toBe(202); // 202 Accepted
            expect(res.body.success).toBe(true);
            expect(res.body.data.id).toBeDefined();
            expect(res.body.data.status).toBe("QUEUED");
            expect(res.body.data.mediaType).toBe("VIDEO");

            videoMediaId = res.body.data.id;
            createdMediaIds.push(videoMediaId);
        });

        it("4. should poll for and verify the completed VIDEO analysis results", async () => {
            const finalMedia = await pollForCompletion(videoMediaId, authToken);

            // Verify final status
            expect(finalMedia.status).toMatch(/ANALYZED|PARTIALLY_ANALYZED/);

            // Verify analysis results
            expect(Array.isArray(finalMedia.analyses)).toBe(true);
            expect(finalMedia.analyses.length).toBeGreaterThan(0);

            const firstAnalysis = finalMedia.analyses[0];
            expect(firstAnalysis.status).toBe("COMPLETED");
            expect(firstAnalysis.prediction).toMatch(/REAL|FAKE/);

            // Verify that video-specific data exists and audio-specific data does not
            expect(firstAnalysis.analysisDetails).toBeDefined();
            expect(firstAnalysis.frameAnalysis.length).toBeGreaterThan(0);
            expect(firstAnalysis.audioAnalysis).toBeNull();
        });
    });

    // --- AUDIO ANALYSIS TEST ---
    describe("Audio Media Type", () => {
        let audioMediaId;

        it("5. should upload an AUDIO file and queue it for analysis", async () => {
            const res = await api
                .post("/api/v1/media") // Use the new generic endpoint
                .set("Authorization", `Bearer ${authToken}`)
                .field("description", "E2E Audio Test")
                .attach("file", TEST_AUDIO_PATH); // Use the new generic field name 'file'

            expect(res.statusCode).toBe(202);
            expect(res.body.success).toBe(true);
            expect(res.body.data.id).toBeDefined();
            expect(res.body.data.status).toBe("QUEUED");
            expect(res.body.data.mediaType).toBe("AUDIO");

            audioMediaId = res.body.data.id;
            createdMediaIds.push(audioMediaId);
        });

        it("6. should poll for and verify the completed AUDIO analysis results", async () => {
            const finalMedia = await pollForCompletion(audioMediaId, authToken);

            // Verify final status
            expect(finalMedia.status).toBe("ANALYZED");

            // Verify analysis results
            expect(Array.isArray(finalMedia.analyses)).toBe(true);
            expect(finalMedia.analyses.length).toBeGreaterThan(0);

            const audioAnalysisResult = finalMedia.analyses.find((a) =>
                a.model.includes("SCATTERING-WAVE-V1")
            );
            expect(audioAnalysisResult).toBeDefined();
            expect(audioAnalysisResult.status).toBe("COMPLETED");

            // Verify that audio-specific data exists
            expect(audioAnalysisResult.audioAnalysis).toBeDefined();
            expect(audioAnalysisResult.audioAnalysis.rmsEnergy).toBeGreaterThan(
                0
            );
            expect(
                audioAnalysisResult.audioAnalysis.spectralCentroid
            ).toBeGreaterThan(0);

            // Verify that video-specific data does NOT exist
            expect(audioAnalysisResult.analysisDetails).toBeNull();
            expect(audioAnalysisResult.frameAnalysis.length).toBe(0);
        });
    });
});
