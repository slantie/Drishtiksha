// src/tests/api/monitoring.integration.test.js

/**
 * @vitest-environment node
 */
import { describe, it, expect, beforeAll } from "vitest";
import request from "supertest";
import { API_BASE_URL } from "../constants/apiEndpoints.js";

const api = request(API_BASE_URL);
const API_VERSION = "/api/v1";

describe("Monitoring API Integration", () => {
    let authToken = "";

    beforeAll(async () => {
        // Create a user to get a valid token for authenticated routes
        const testUser = {
            email: `monitor-user-${Date.now()}@supertest.com`,
            password: "password123",
            firstName: "Monitor",
            lastName: "Test",
        };
        await api.post(`${API_VERSION}/auth/signup`).send(testUser);
        const loginRes = await api
            .post(`${API_VERSION}/auth/login`)
            .send(testUser);
        authToken = loginRes.body.data.token;
    });

    it("GET /api/v1/monitoring/server-status - should retrieve the ML server status", async () => {
        const response = await api
            .get(`${API_VERSION}/monitoring/server-status`)
            .set("Authorization", `Bearer ${authToken}`)
            .expect(200);

        expect(response.body.success).toBe(true);
        const data = response.body.data;
        expect(data.serviceName).toBeDefined();
        expect(data.deviceInfo).toBeDefined();
        expect(data.systemInfo).toBeDefined();
    }, 20000); // Increased timeout for external service call

    it("GET /api/v1/monitoring/queue-status - should retrieve the job queue status", async () => {
        const response = await api
            .get(`${API_VERSION}/monitoring/queue-status`)
            .set("Authorization", `Bearer ${authToken}`)
            .expect(200);

        expect(response.body.success).toBe(true);
        const data = response.body.data;
        expect(data).toHaveProperty("pendingJobs");
        expect(data).toHaveProperty("activeJobs");
        expect(typeof data.completedJobs).toBe("number");
    });
});
