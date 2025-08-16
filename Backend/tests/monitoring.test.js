// tests/monitoring.test.js

import request from "supertest";
import { app } from "../src/app.js";
import prisma from "../src/config/database.js";

describe("Monitoring Endpoints", () => {
    let api;
    let authToken;
    const testUser = {
        email: `monitor-user-${Date.now()}@example.com`,
        password: "password123",
        firstName: "Monitor",
        lastName: "Test",
    };

    beforeAll(async () => {
        api = request(app);
        const registerRes = await api
            .post("/api/v1/auth/signup")
            .send(testUser);
        authToken = registerRes.body.data.token;
    });

    afterAll(async () => {
        await prisma.user.deleteMany({ where: { email: testUser.email } });
    });

    it("GET /api/v1/monitoring/server-status - should return detailed server statistics", async () => {
        const res = await api
            .get("/api/v1/monitoring/server-status")
            .set("Authorization", `Bearer ${authToken}`);

        expect(res.statusCode).toBe(200);
        expect(res.body.success).toBe(true);
        const { data } = res.body;

        // CORRECTED: Use camelCase properties to match the API's transformed response.
        expect(data.serviceName).toBeDefined();
        expect(data.deviceInfo).toBeDefined();
        expect(data.systemInfo).toBeDefined();
        expect(data.modelsInfo).toBeInstanceOf(Array);

        expect(typeof data.deviceInfo.type).toBe("string");
        expect(typeof data.systemInfo.pythonVersion).toBe("string");
        if (data.modelsInfo.length > 0) {
            expect(data.modelsInfo[0].name).toBeDefined();
        }
    }, 25000);

    it("GET /api/v1/monitoring/queue-status - should return the status of the job queue", async () => {
        const res = await api
            .get("/api/v1/monitoring/queue-status")
            .set("Authorization", `Bearer ${authToken}`);

        expect(res.statusCode).toBe(200);
        expect(res.body.success).toBe(true);
        const { data } = res.body;

        expect(data).toHaveProperty("pendingJobs");
        expect(typeof data.pendingJobs).toBe("number");
        expect(data).toHaveProperty("activeJobs");
        expect(typeof data.activeJobs).toBe("number");
        expect(data).toHaveProperty("completedJobs");
        expect(typeof data.completedJobs).toBe("number");
    });

    it("GET /api/v1/monitoring/stats/analysis - should return historical analysis stats", async () => {
        const res = await api
            .get("/api/v1/monitoring/stats/analysis?timeframe=24h")
            .set("Authorization", `Bearer ${authToken}`);

        expect(res.statusCode).toBe(200);
        expect(res.body.success).toBe(true);
        const { data } = res.body;

        expect(data.timeframe).toBe("24h");
        expect(data).toHaveProperty("total");
        expect(data).toHaveProperty("successful");
        expect(data).toHaveProperty("failed");
        expect(data).toHaveProperty("avgProcessingTime");
    });
});
