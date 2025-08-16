// src/tests/api/videos.integration.test.js

/**
 * @vitest-environment node
 */
import { describe, it, expect, beforeAll } from "vitest";
import request from "supertest";
import path from "path";
import { API_BASE_URL } from "../constants/apiEndpoints.js";

const api = request(API_BASE_URL);
const API_VERSION = "/api/v1";
const VIDEO_PATH = path.resolve(
    "D:\\Slantie@LDRP\\7th Semester\\NTRO\\Website\\Frontend\\assets\\test-video.mp4"
);

describe.sequential("Videos API Integration", () => {
    let authToken = "";
    let newVideoId = "";

    beforeAll(async () => {
        const testUser = {
            email: `video-user-${Date.now()}@supertest.com`,
            password: "password123",
            firstName: "Video",
            lastName: "Test",
        };
        await api.post(`${API_VERSION}/auth/signup`).send(testUser);
        const loginRes = await api
            .post(`${API_VERSION}/auth/login`)
            .send(testUser);
        authToken = loginRes.body.data.token;
    });

    it("POST /api/v1/videos - should upload a video and queue it for analysis", async () => {
        const response = await api
            .post(`${API_VERSION}/videos`)
            .set("Authorization", `Bearer ${authToken}`)
            .field("description", "Supertest video upload")
            .attach("video", VIDEO_PATH)
            .expect(202); // 202 Accepted is the correct code for queuing

        expect(response.body.success).toBe(true);
        const video = response.body.data;
        expect(video.id).toBeDefined();
        expect(video.status).toBe("QUEUED");
        expect(video.filename).toBe("test-video.mp4");
        newVideoId = video.id;
    }, 30000); // 30-second timeout for upload

    it("GET /api/v1/videos - should retrieve the list of videos including the new one", async () => {
        const response = await api
            .get(`${API_VERSION}/videos`)
            .set("Authorization", `Bearer ${authToken}`)
            .expect(200);

        expect(response.body.success).toBe(true);
        const videos = response.body.data;
        const uploadedVideo = videos.find((v) => v.id === newVideoId);
        expect(uploadedVideo).toBeDefined();
    });

    it("GET /api/v1/videos/:id - should retrieve the specific uploaded video", async () => {
        const response = await api
            .get(`${API_VERSION}/videos/${newVideoId}`)
            .set("Authorization", `Bearer ${authToken}`)
            .expect(200);

        expect(response.body.success).toBe(true);
        expect(response.body.data.id).toBe(newVideoId);
    });

    it("DELETE /api/v1/videos/:id - should delete the uploaded video", async () => {
        const response = await api
            .delete(`${API_VERSION}/videos/${newVideoId}`)
            .set("Authorization", `Bearer ${authToken}`)
            .expect(200);

        expect(response.body.success).toBe(true);
        expect(response.body.message).toBe("Video deleted successfully.");

        // Verify it's gone
        await api
            .get(`${API_VERSION}/videos/${newVideoId}`)
            .set("Authorization", `Bearer ${authToken}`)
            .expect(404);
    });
});
