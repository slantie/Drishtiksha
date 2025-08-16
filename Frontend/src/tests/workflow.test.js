// src/tests/api/workflow.integration.test.js

/**
 * @vitest-environment node
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import request from "supertest";
import path from "path";
import { io } from "socket.io-client";
import { API_BASE_URL } from "../constants/apiEndpoints";

const api = request(API_BASE_URL);
const API_VERSION = "/api/v1";
const VIDEO_PATH = path.resolve(
    "D:\\Slantie@LDRP\\7th Semester\\NTRO\\Website\\Frontend\\assets\\test-video.mp4"
);

describe("Full E2E Video Processing Workflow", () => {
    let authToken = "";
    let newVideoId = "";
    let socket;

    beforeAll(async () => {
        const testUser = {
            email: `workflow-user-${Date.now()}@supertest.com`,
            password: "password123",
            firstName: "Workflow",
            lastName: "Test",
        };
        await api.post(`${API_VERSION}/auth/signup`).send(testUser);
        const loginRes = await api
            .post(`${API_VERSION}/auth/login`)
            .send(testUser);
        authToken = loginRes.body.data.token;
    });

    afterAll(() => {
        if (socket && socket.connected) {
            socket.disconnect();
        }
    });

    it("should upload a video, receive real-time updates, and confirm analysis is complete", async () => {
        // --- 1. Set up Socket.io Client and Promise ---
        const receivedEvents = [];
        const finalStatusPromise = new Promise((resolve, reject) => {
            socket = io(API_BASE_URL, {
                auth: { token: authToken },
                transports: ["websocket"],
            });

            socket.on("connect", () =>
                console.log("[TestClient] Socket connected.")
            );

            // **CORRECTED LOGIC:**
            // We now accept a wider range of valid progress events.
            socket.on("progress_update", (data) => {
                if (data.videoId === newVideoId) {
                    console.log(`[TestClient] Received event: ${data.event}`);
                    receivedEvents.push(data.event);

                    // Define all valid event types the backend can send
                    const VALID_EVENTS = [
                        "PROCESSING_STARTED",
                        "ANALYSIS_STARTED",
                        "FRAME_ANALYSIS_PROGRESS",
                        "ANALYSIS_COMPLETED",
                        "VISUALIZATION_UPLOADING",
                        "VISUALIZATION_COMPLETED",
                    ];
                    // This assertion will now correctly handle all event types
                    expect(VALID_EVENTS).toContain(data.event);
                }
            });

            socket.on("video_update", (finalVideo) => {
                if (finalVideo.id === newVideoId) {
                    console.log(
                        `[TestClient] Received final video_update with status: ${finalVideo.status}`
                    );
                    resolve(finalVideo);
                }
            });

            socket.on("processing_error", (errorData) => {
                if (errorData.videoId === newVideoId) {
                    reject(new Error(errorData.error));
                }
            });

            socket.on("connect_error", (err) => {
                reject(new Error(`Socket connection error: ${err.message}`));
            });
        });

        // --- 2. Upload the Video ---
        const response = await api
            .post(`${API_VERSION}/videos`)
            .set("Authorization", `Bearer ${authToken}`)
            .field("description", "E2E workflow test")
            .attach("video", VIDEO_PATH)
            .expect(202);

        newVideoId = response.body.data.id;
        console.log(
            `[Test] Video uploaded with ID: ${newVideoId}. Waiting for analysis...`
        );

        // --- 3. Wait for the final confirmation ---
        const finalVideoState = await finalStatusPromise;

        // --- 4. Assert the Final State ---
        expect(finalVideoState).toBeDefined();
        expect(["ANALYZED", "PARTIALLY_ANALYZED"]).toContain(
            finalVideoState.status
        );
        expect(finalVideoState.analyses.length).toBeGreaterThan(0);

        const firstAnalysis = finalVideoState.analyses[0];
        expect(firstAnalysis.status).toBe("COMPLETED");
        expect(firstAnalysis.prediction).toBeDefined();

        // **NEW ASSERTION:**
        // Verify that we received real-time progress from the Python server.
        expect(receivedEvents).toContain("FRAME_ANALYSIS_PROGRESS");
        console.log(
            "[Test] Successfully received frame analysis progress events."
        );
    }, 300000);
});
