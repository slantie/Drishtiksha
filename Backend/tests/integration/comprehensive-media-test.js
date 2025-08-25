/**
 * Comprehensive Media Analysis Integration Test
 * Tests both VIDEO and AUDIO analysis with real ML server
 */

import { describe, test, expect, beforeAll, afterAll } from "@jest/globals";
import request from "supertest";
import fs from "fs";
import path from "path";
import { app } from "../../src/app.js";
import { modelAnalysisService } from "../../src/services/modelAnalysis.service.js";
import "../setup/testSetup.js";

describe("Comprehensive Media Analysis Integration", () => {
    let testUser;
    let authToken;
    let serverStats;

    // Test files
    const testVideoPath = path.join(
        process.cwd(),
        "tests",
        "fixtures",
        "test-video.mp4"
    );
    const testAudioPath = path.join(
        process.cwd(),
        "tests",
        "fixtures",
        "test-audio.mp3"
    );

    beforeAll(async () => {
        // Verify server connection
        serverStats = await modelAnalysisService.getServerStatistics();
        console.log(
            `✅ Server connected with ${serverStats.active_models_count} models`
        );
        console.log(
            `📋 Models: ${serverStats.configuration.active_models.join(", ")}`
        );

        // Create test user using the same pattern as working tests
        const timestamp = Date.now();
        const userData = {
            email: `comprehensive-test-${timestamp}@example.com`,
            password: "ComprehensiveTest123!",
            firstName: "Comprehensive",
            lastName: "Test",
        };

        const signupRes = await request(app)
            .post("/api/v1/auth/signup")
            .send(userData);

        expect(signupRes.statusCode).toBe(201);

        const loginRes = await request(app).post("/api/v1/auth/login").send({
            email: userData.email,
            password: userData.password,
        });

        expect(loginRes.statusCode).toBe(200);
        authToken = loginRes.body.data.token;
        testUser = { email: userData.email, id: loginRes.body.data.user?.id };
        console.log(`✅ Test user authenticated: ${testUser.email}`);
    });

    afterAll(async () => {
        // Cleanup test user - they will cleanup their own media
        console.log(`🧹 Cleaning up test user: ${testUser?.email}`);
    });

    test("should analyze VIDEO file with 4 working models", async () => {
        console.log(`🎥 Testing VIDEO analysis with file: ${testVideoPath}`);

        // Verify test file exists
        expect(fs.existsSync(testVideoPath)).toBe(true);

        // Upload video
        const uploadRes = await request(app)
            .post("/api/v1/media")
            .set("Authorization", `Bearer ${authToken}`)
            .field("description", "Comprehensive video analysis test")
            .attach("file", testVideoPath);

        expect(uploadRes.statusCode).toBe(202); // 202 Accepted for processing
        expect(uploadRes.body.data.mediaType).toBe("VIDEO");

        const mediaId = uploadRes.body.data.id;
        console.log(`✅ Video uploaded: ${mediaId}`);

        // Wait for analysis completion
        const finalMedia = await waitForAnalysisCompletion(mediaId, authToken);

        console.log(`📊 Final Status: ${finalMedia.status}`);
        console.log(`📈 Completed Analyses: ${finalMedia.analyses.length}`);

        // Expected: 4 successful analyses (all except SCATTERING-WAVE-V1 which needs audio)
        expect(["ANALYZED", "PARTIALLY_ANALYZED"]).toContain(finalMedia.status);
        expect(finalMedia.analyses.length).toBeGreaterThanOrEqual(4);

        // Check that video models completed successfully
        const videoModels = [
            "SIGLIP-LSTM-V4",
            "COLOR-CUES-LSTM-V1",
            "EFFICIENTNET-B7-V1",
            "EYEBLINK-CNN-LSTM-V1",
        ];
        const completedVideoModels = finalMedia.analyses
            .filter(
                (a) => a.status === "COMPLETED" && videoModels.includes(a.model)
            )
            .map((a) => a.model);

        console.log(
            `✅ Completed Video Models: ${completedVideoModels.join(", ")}`
        );
        expect(completedVideoModels.length).toBe(4);

        // Check for audio model failure (expected)
        const audioModelAnalysis = finalMedia.analyses.find(
            (a) => a.model === "SCATTERING-WAVE-V1"
        );
        if (audioModelAnalysis) {
            console.log(
                `❌ Audio model status: ${audioModelAnalysis.status} (expected failure - no audio track)`
            );
            expect(audioModelAnalysis.status).toBe("FAILED");
        }
    }, 300000); // 5 minute timeout

    test("should analyze AUDIO file with audio model", async () => {
        console.log(`🎵 Testing AUDIO analysis with file: ${testAudioPath}`);

        // Verify test file exists
        expect(fs.existsSync(testAudioPath)).toBe(true);

        // Upload audio
        const uploadRes = await request(app)
            .post("/api/v1/media")
            .set("Authorization", `Bearer ${authToken}`)
            .field("description", "Comprehensive audio analysis test")
            .attach("file", testAudioPath);

        expect(uploadRes.statusCode).toBe(202); // 202 Accepted for processing
        expect(uploadRes.body.data.mediaType).toBe("AUDIO");

        const mediaId = uploadRes.body.data.id;
        console.log(`✅ Audio uploaded: ${mediaId}`);

        // Wait for analysis completion
        const finalMedia = await waitForAnalysisCompletion(mediaId, authToken);

        console.log(`📊 Final Status: ${finalMedia.status}`);
        console.log(`📈 Completed Analyses: ${finalMedia.analyses.length}`);

        // Expected: 1 successful analysis (SCATTERING-WAVE-V1)
        expect(["ANALYZED", "PARTIALLY_ANALYZED"]).toContain(finalMedia.status);
        expect(finalMedia.analyses.length).toBeGreaterThanOrEqual(1);

        // Check that audio model completed successfully
        const audioModelAnalysis = finalMedia.analyses.find(
            (a) => a.model === "SCATTERING-WAVE-V1"
        );
        expect(audioModelAnalysis).toBeDefined();
        expect(audioModelAnalysis.status).toBe("COMPLETED");
        expect(audioModelAnalysis.prediction).toMatch(/^(REAL|FAKE)$/);
        expect(audioModelAnalysis.confidence).toBeGreaterThan(0);

        console.log(
            `✅ Audio Analysis Result: ${audioModelAnalysis.prediction} (${audioModelAnalysis.confidence})`
        );

        // Check for video model failures (expected for audio file)
        const videoModels = [
            "SIGLIP-LSTM-V4",
            "COLOR-CUES-LSTM-V1",
            "EFFICIENTNET-B7-V1",
            "EYEBLINK-CNN-LSTM-V1",
        ];
        const failedVideoModels = finalMedia.analyses
            .filter(
                (a) => a.status === "FAILED" && videoModels.includes(a.model)
            )
            .map((a) => a.model);

        console.log(
            `❌ Failed Video Models on Audio: ${failedVideoModels.join(
                ", "
            )} (expected)`
        );
    }, 300000); // 5 minute timeout

    test("should verify server statistics and model availability", async () => {
        console.log(`📊 Verifying server configuration`);

        expect(serverStats.status).toBe("running");
        expect(serverStats.active_models_count).toBe(5);
        expect(serverStats.device_info.type).toBe("cuda");
        expect(serverStats.device_info.name).toContain("RTX 4050");

        // Verify all expected models are loaded
        const expectedModels = [
            "SIGLIP-LSTM-V4",
            "COLOR-CUES-LSTM-V1",
            "EFFICIENTNET-B7-V1",
            "EYEBLINK-CNN-LSTM-V1",
            "SCATTERING-WAVE-V1",
        ];

        const loadedModels = serverStats.models_info
            .filter((m) => m.loaded)
            .map((m) => m.name);

        expectedModels.forEach((model) => {
            expect(loadedModels).toContain(model);
        });

        console.log(
            `✅ All ${expectedModels.length} models verified as loaded`
        );
    });
});

/**
 * Wait for analysis completion with timeout
 */
async function waitForAnalysisCompletion(
    mediaId,
    authToken,
    maxWaitTime = 300000
) {
    const startTime = Date.now();
    const checkInterval = 2000;

    while (Date.now() - startTime < maxWaitTime) {
        const mediaRes = await request(app)
            .get(`/api/v1/media/${mediaId}`)
            .set("Authorization", `Bearer ${authToken}`);

        expect(mediaRes.statusCode).toBe(200);
        const media = mediaRes.body.data;

        console.log(
            `📊 Status: ${media.status}, Analyses: ${media.analyses.length}`
        );

        // Check if analysis is complete
        if (
            ["ANALYZED", "PARTIALLY_ANALYZED", "FAILED"].includes(media.status)
        ) {
            return media;
        }

        await new Promise((resolve) => setTimeout(resolve, checkInterval));
    }

    throw new Error(`Analysis did not complete within ${maxWaitTime}ms`);
}
