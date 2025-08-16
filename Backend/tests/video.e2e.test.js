// tests/video.e2e.test.js

import request from "supertest";
import path from "path";
import { io as Client } from "socket.io-client";
import { app } from "../src/app.js";
import prisma from "../src/config/database.js";

describe("Video Analysis E2E Workflow", () => {
    let api;
    let authToken;
    let userId;
    let socket;

    const TEST_VIDEO_PATH = path.join(
        process.cwd(),
        "tests",
        "fixtures",
        "test-video.mp4"
    );
    const testUser = {
        email: `e2e-user-${Date.now()}@example.com`,
        password: "password123",
        firstName: "E2E",
        lastName: "Test",
    };

    beforeAll(async () => {
        api = request(app);
        const registerRes = await api
            .post("/api/v1/auth/signup")
            .send(testUser);
        authToken = registerRes.body.data.token;
        userId = registerRes.body.data.user.id;
    });

    afterAll(async () => {
        if (socket && socket.connected) {
            socket.disconnect();
        }
        if (userId) {
            // Prisma's cascade delete on the User model will handle cleanup
            await prisma.user.delete({ where: { id: userId } });
        }
    });

    it("should upload a video, receive real-time updates, and verify final analysis", async () => {
        let finalVideoState = null;

        socket = Client(`http://localhost:${process.env.PORT || 4000}`, {
            auth: { token: authToken },
        });

        // REASON: This Promise-based approach is more robust for testing async events.
        // It will wait for the 'video_update' event that signals a final status.
        const finalResultPromise = new Promise((resolve, reject) => {
            socket.on("connect", () => {
                console.log("Test client connected via WebSocket.");
            });

            socket.on("video_update", (data) => {
                console.log(
                    `Test client received 'video_update' event with status: ${data.status}`
                );
                // Resolve the promise only when we get a terminal status.
                if (
                    ["ANALYZED", "PARTIALLY_ANALYZED", "FAILED"].includes(
                        data.status
                    )
                ) {
                    finalVideoState = data;
                    resolve(data);
                }
            });

            socket.on("processing_error", (data) => {
                console.error(
                    "Test client received 'processing_error' event:",
                    data
                );
                reject(new Error(`Processing failed: ${data.error}`));
            });
        });

        const uploadRes = await api
            .post("/api/v1/videos")
            .set("Authorization", `Bearer ${authToken}`)
            .field("description", "E2E Test")
            .attach("video", TEST_VIDEO_PATH);

        expect(uploadRes.statusCode).toBe(202);

        // Wait for the final result from the WebSocket.
        await finalResultPromise;

        // Step 4: Validate the final results.
        expect(finalVideoState).not.toBeNull();
        expect(finalVideoState.status).toMatch(/ANALYZED|PARTIALLY_ANALYZED/);
        expect(finalVideoState.analyses.length).toBeGreaterThan(0);

        const firstAnalysis = finalVideoState.analyses[0];
        expect(firstAnalysis.status).toBe("COMPLETED");
        expect(firstAnalysis.prediction).toBeDefined();

        expect(firstAnalysis.modelInfo).not.toBeNull();
        expect(firstAnalysis.systemInfo).not.toBeNull();
        expect(firstAnalysis.modelInfo.modelName).toBeDefined();
        expect(firstAnalysis.systemInfo.processingDevice).toBeDefined();
    }, 300000); // 5-minute timeout to allow for full processing.
});
