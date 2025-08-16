// src/tests/api/auth.integration.test.js

/**
 * @vitest-environment node
 */
import { describe, it, expect } from "vitest";
import request from "supertest";
import { API_BASE_URL } from "../constants/apiEndpoints.js";

const api = request(API_BASE_URL);
const API_VERSION = "/api/v1";

describe.sequential("Auth API Integration", () => {
    const testUser = {
        email: `test-user-${Date.now()}@supertest.com`,
        password: "password123",
        firstName: "Supertest",
        lastName: "User",
    };
    let authToken = "";

    it("POST /api/v1/auth/signup - should register a new user successfully", async () => {
        const response = await api
            .post(`${API_VERSION}/auth/signup`)
            .send(testUser)
            .expect(201); // Assert HTTP status code

        expect(response.body.success).toBe(true);
        expect(response.body.data.user.email).toBe(testUser.email);
        expect(response.body.data.user).not.toHaveProperty("password");
    });

    it("POST /api/v1/auth/login - should log in the user and return a token", async () => {
        const response = await api
            .post(`${API_VERSION}/auth/login`)
            .send({ email: testUser.email, password: testUser.password })
            .expect(200);

        expect(response.body.success).toBe(true);
        expect(response.body.data.token).toBeDefined();
        authToken = response.body.data.token; // Save the token for subsequent tests
    });

    it("GET /api/v1/auth/profile - should fetch the user profile with a valid token", async () => {
        expect(authToken).toBeDefined(); // Ensure token was received

        const response = await api
            .get(`${API_VERSION}/auth/profile`)
            .set("Authorization", `Bearer ${authToken}`)
            .expect(200);

        expect(response.body.success).toBe(true);
        expect(response.body.data.email).toBe(testUser.email);
    });

    it("GET /api/v1/auth/profile - should fail without a token", async () => {
        await api.get(`${API_VERSION}/auth/profile`).expect(401);
    });
});
