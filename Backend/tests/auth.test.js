// tests/auth.test.js

import request from "supertest";
import { app } from "../src/app.js";
import prisma from "../src/config/database.js";

describe("Authentication Endpoints", () => {
    const testUser = {
        email: `testuser-${Date.now()}@example.com`,
        password: "password123",
        firstName: "Test",
        lastName: "User",
    };

    // Clean up the created user after all tests in this file have run.
    afterAll(async () => {
        await prisma.user.deleteMany({
            where: { email: testUser.email },
        });
    });

    it("should sign up a new user successfully", async () => {
        const res = await request(app)
            .post("/api/v1/auth/signup")
            .send(testUser);

        // Verify the response structure and status code
        expect(res.statusCode).toEqual(201);
        expect(res.body.success).toBe(true);
        expect(res.body.data.user.email).toBe(testUser.email);
        expect(res.body.data.token).toBeDefined();
        // Ensure the password is not returned
        expect(res.body.data.user.password).toBeUndefined();
    });

    it("should log in an existing user and return a token", async () => {
        const res = await request(app).post("/api/v1/auth/login").send({
            email: testUser.email,
            password: testUser.password,
        });

        expect(res.statusCode).toEqual(200);
        expect(res.body.success).toBe(true);
        expect(res.body.data.user.email).toBe(testUser.email);
        expect(res.body.data.token).toBeDefined();
    });

    it("should fail to log in with an incorrect password", async () => {
        const res = await request(app).post("/api/v1/auth/login").send({
            email: testUser.email,
            password: "wrongpassword",
        });

        expect(res.statusCode).toEqual(401);
        expect(res.body.success).toBe(false);
        expect(res.body.message).toBe("Invalid email or password");
    });
});
