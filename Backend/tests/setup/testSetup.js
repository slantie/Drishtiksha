/**
 * Test Setup and Configuration
 *
 * Global test configuration, mocks, and utilities for the enhanced media analysis API tests
 * This file now integrates with real environment variables for proper backend-server integration testing
 */

import { jest } from "@jest/globals";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import TEST_CONFIG, { testUtils } from "./testEnv.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Enhanced test configuration merging environment and defaults
export { TEST_CONFIG, testUtils };

// Legacy test config for backward compatibility
export const LEGACY_TEST_CONFIG = {
    timeout: TEST_CONFIG.defaultTimeout,
    apiUrl: "http://localhost:3000/api",
    testUserId: "test-user-123",
    testVideoId: "test-video-123",
    mockServerUrl: TEST_CONFIG.serverUrl,
    testDataPath: join(__dirname, "../fixtures"),
};

// Only mock environment variables if they're not already set for integration tests
if (!process.env.SERVER_URL || process.env.NODE_ENV === "test-unit") {
    console.log("🧪 Setting up mocked environment for unit tests");
    process.env.SERVER_URL = process.env.SERVER_URL || "http://localhost:8000";
    process.env.SERVER_API_KEY = process.env.SERVER_API_KEY || "test-api-key";
    process.env.JWT_SECRET = process.env.JWT_SECRET || "test-jwt-secret";
    if (!process.env.DATABASE_URL) {
        process.env.DATABASE_URL = "file:./test.db";
    }
} else {
    console.log(
        "🔗 Using real environment configuration for integration tests"
    );
}

// Global mocks setup
export const setupGlobalMocks = () => {
    // Mock console methods to reduce noise in tests
    global.console = {
        ...console,
        log: jest.fn(),
        debug: jest.fn(),
        info: jest.fn(),
        warn: jest.fn(),
        error: jest.fn(),
    };

    // Mock file system operations
    jest.mock("fs", () => ({
        existsSync: jest.fn(),
        createReadStream: jest.fn(),
        createWriteStream: jest.fn(),
        mkdirSync: jest.fn(),
        writeFileSync: jest.fn(),
        readFileSync: jest.fn(),
        unlinkSync: jest.fn(),
        promises: {
            mkdir: jest.fn(),
            unlink: jest.fn(),
            writeFile: jest.fn(),
            readFile: jest.fn(),
        },
    }));

    // Mock axios for HTTP requests
    jest.mock("axios", () => ({
        default: jest.fn(),
        get: jest.fn(),
        post: jest.fn(),
        create: jest.fn(() => ({
            get: jest.fn(),
            post: jest.fn(),
            interceptors: {
                request: { use: jest.fn() },
                response: { use: jest.fn() },
            },
        })),
    }));

    // Mock Prisma client
    jest.mock("@prisma/client", () => ({
        PrismaClient: jest.fn().mockImplementation(() => ({
            video: {
                create: jest.fn(),
                findUnique: jest.fn(),
                findMany: jest.fn(),
                update: jest.fn(),
                delete: jest.fn(),
            },
            deepfakeAnalysis: {
                create: jest.fn(),
                findUnique: jest.fn(),
                findFirst: jest.fn(),
                findMany: jest.fn(),
                update: jest.fn(),
                updateMany: jest.fn(),
                delete: jest.fn(),
            },
            user: {
                create: jest.fn(),
                findUnique: jest.fn(),
                findMany: jest.fn(),
                update: jest.fn(),
            },
            $connect: jest.fn(),
            $disconnect: jest.fn(),
            $transaction: jest.fn(),
        })),
    }));

    // Mock Cloudinary
    jest.mock("cloudinary", () => ({
        v2: {
            config: jest.fn(),
            uploader: {
                upload: jest.fn(),
                upload_stream: jest.fn(),
                destroy: jest.fn(),
            },
        },
    }));

    // Mock multer
    jest.mock("multer", () => ({
        default: jest.fn(() => ({
            single: jest.fn(() => (req, res, next) => {
                req.file = {
                    originalname: "test-video.mp4",
                    mimetype: "video/mp4",
                    size: 1024 * 1024,
                    buffer: Buffer.from("mock-video-data"),
                    path: "/tmp/test-video.mp4",
                };
                next();
            }),
            array: jest.fn(),
            fields: jest.fn(),
        })),
        memoryStorage: jest.fn(),
        diskStorage: jest.fn(),
    }));

    // Mock Winston logger
    jest.mock("winston", () => ({
        createLogger: jest.fn(() => ({
            info: jest.fn(),
            warn: jest.fn(),
            error: jest.fn(),
            debug: jest.fn(),
        })),
        format: {
            combine: jest.fn(),
            timestamp: jest.fn(),
            printf: jest.fn(),
            colorize: jest.fn(),
            simple: jest.fn(),
        },
        transports: {
            Console: jest.fn(),
            File: jest.fn(),
        },
    }));
};

// Test data factories
export const createTestUser = (overrides = {}) => ({
    id: TEST_CONFIG.testUserId,
    email: "test@example.com",
    firstName: "Test",
    lastName: "User",
    role: "USER",
    createdAt: new Date(),
    updatedAt: new Date(),
    ...overrides,
});

export const createTestVideo = (overrides = {}) => ({
    id: TEST_CONFIG.testVideoId,
    filename: "test-video.mp4",
    mimetype: "video/mp4",
    size: 1024 * 1024,
    description: "Test video for analysis",
    url: "https://example.com/test-video.mp4",
    publicId: "test-public-id",
    localPath: "/tmp/test-video.mp4",
    userId: TEST_CONFIG.testUserId,
    status: "UPLOADED",
    createdAt: new Date(),
    updatedAt: new Date(),
    analyses: [],
    ...overrides,
});

export const createTestAnalysis = (overrides = {}) => ({
    id: "test-analysis-123",
    videoId: TEST_CONFIG.testVideoId,
    model: "SIGLIP_LSTM_V1",
    type: "QUICK",
    status: "COMPLETED",
    confidence: 0.85,
    isDeepfake: false,
    prediction: "REAL",
    processingStartedAt: new Date(Date.now() - 5000),
    processingCompletedAt: new Date(),
    processingTime: 2.5,
    modelVersion: "1.0.0",
    rawResult: {
        prediction: "REAL",
        confidence: 0.85,
        processing_time: 2.5,
    },
    createdAt: new Date(),
    updatedAt: new Date(),
    analysisDetails: null,
    frameAnalyses: [],
    temporalAnalyses: [],
    modelInfo: null,
    systemInfo: null,
    analysisError: null,
    ...overrides,
});

export const createDetailedAnalysis = (overrides = {}) =>
    createTestAnalysis({
        type: "DETAILED",
        analysisDetails: {
            id: "details-123",
            analysisId: "test-analysis-123",
            frameConsistency: 0.85,
            temporalCoherence: 0.9,
            facialArtifacts: 0.15,
            additionalMetrics: {
                blurriness: 0.1,
                compression_artifacts: 0.05,
                lighting_consistency: 0.95,
            },
            createdAt: new Date(),
        },
        ...overrides,
    });

export const createFramesAnalysis = (overrides = {}) =>
    createTestAnalysis({
        type: "FRAMES",
        frameAnalyses: [
            {
                id: "frame-1",
                analysisId: "test-analysis-123",
                frameNumber: 0,
                confidence: 0.85,
                prediction: "REAL",
                isDeepfake: false,
                additionalData: { timestamp: 0.0 },
                createdAt: new Date(),
            },
            {
                id: "frame-2",
                analysisId: "test-analysis-123",
                frameNumber: 10,
                confidence: 0.82,
                prediction: "REAL",
                isDeepfake: false,
                additionalData: { timestamp: 0.33 },
                createdAt: new Date(),
            },
            {
                id: "frame-3",
                analysisId: "test-analysis-123",
                frameNumber: 20,
                confidence: 0.78,
                prediction: "REAL",
                isDeepfake: false,
                additionalData: { timestamp: 0.66 },
                createdAt: new Date(),
            },
        ],
        ...overrides,
    });

export const createVisualizationAnalysis = (overrides = {}) =>
    createTestAnalysis({
        type: "VISUALIZE",
        confidence: 1.0,
        prediction: "VISUALIZATION",
        visualizedUrl: "/uploads/visualizations/visualization-123456789.mp4",
        ...overrides,
    });

// Mock server responses
export const createMockServerResponse = (type, success = true) => {
    const baseResponse = {
        success,
        timestamp: new Date().toISOString(),
        processing_time: 2.5,
        model: "siglip_lstm_v1",
    };

    switch (type) {
        case "QUICK":
            return {
                ...baseResponse,
                prediction: "REAL",
                confidence: 0.85,
                is_deepfake: false,
            };

        case "DETAILED":
            return {
                ...baseResponse,
                prediction: "REAL",
                confidence: 0.85,
                is_deepfake: false,
                detailed_metrics: {
                    frame_consistency: 0.85,
                    temporal_coherence: 0.9,
                    facial_artifacts: 0.15,
                },
            };

        case "FRAMES":
            return {
                ...baseResponse,
                prediction: "REAL",
                confidence: 0.85,
                is_deepfake: false,
                frame_analyses: [
                    { frame_number: 0, confidence: 0.85, prediction: "REAL" },
                    { frame_number: 10, confidence: 0.82, prediction: "REAL" },
                    { frame_number: 20, confidence: 0.78, prediction: "REAL" },
                ],
            };

        case "VISUALIZE":
            return {
                success: true,
                visualizationPath: "/tmp/visualization-123.mp4",
                visualizationUrl:
                    "/uploads/visualizations/visualization-123.mp4",
                message: "Visualization generated successfully",
            };

        case "HEALTH":
            return {
                status: "healthy",
                models_loaded: 3,
                memory_usage: "2.1GB",
                uptime: "1h 30m",
                version: "2.0.0",
            };

        case "MODEL_INFO":
            return {
                server_version: "2.0.0",
                available_models: [
                    "siglip_lstm_v1",
                    "siglip_lstm_v3",
                    "color_cues_lstm_v1",
                ],
                model_details: {
                    siglip_lstm_v1: { version: "1.0.0", loaded: true },
                    siglip_lstm_v3: { version: "3.0.0", loaded: true },
                    color_cues_lstm_v1: { version: "1.0.0", loaded: true },
                },
            };

        default:
            return baseResponse;
    }
};

// Helper function to create mock JWT token
export const createMockJWTToken = (payload = {}) => {
    const defaultPayload = {
        id: TEST_CONFIG.testUserId,
        email: "test@example.com",
        role: "USER",
        iat: Math.floor(Date.now() / 1000),
        exp: Math.floor(Date.now() / 1000) + 3600, // 1 hour
    };

    // In real implementation, this would use jsonwebtoken
    const mockToken = Buffer.from(
        JSON.stringify({ ...defaultPayload, ...payload })
    ).toString("base64");
    return `mock.jwt.${mockToken}`;
};

// Test database utilities
export const setupTestDatabase = async () => {
    // Mock database setup for tests
    console.log("Setting up test database...");

    // In a real implementation, this would:
    // 1. Create a test database
    // 2. Run migrations
    // 3. Seed test data

    return Promise.resolve();
};

export const cleanupTestDatabase = async () => {
    // Mock database cleanup
    console.log("Cleaning up test database...");

    // In a real implementation, this would:
    // 1. Clear test data
    // 2. Reset sequences
    // 3. Close connections

    return Promise.resolve();
};

// API testing utilities
export const createApiRequest = (
    method,
    endpoint,
    data = null,
    headers = {}
) => {
    return {
        method: method.toUpperCase(),
        url: `${TEST_CONFIG.apiUrl}${endpoint}`,
        data,
        headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${createMockJWTToken()}`,
            ...headers,
        },
    };
};

export const mockMulterFile = (
    filename = "test-video.mp4",
    mimetype = "video/mp4"
) => ({
    fieldname: "video",
    originalname: filename,
    encoding: "7bit",
    mimetype,
    size: 1024 * 1024, // 1MB
    destination: "/tmp",
    filename: `test-${Date.now()}.mp4`,
    path: `/tmp/test-${Date.now()}.mp4`,
    buffer: Buffer.from("mock-video-data"),
});

// Error response helpers
export const createErrorResponse = (status, message, errors = []) => ({
    success: false,
    error: {
        status,
        message,
        errors,
        timestamp: new Date().toISOString(),
    },
});

// Success response helpers
export const createSuccessResponse = (data, message = "Success") => ({
    success: true,
    data,
    message,
    timestamp: new Date().toISOString(),
});

// Export setup function to be called in test files
export const initializeTestEnvironment = () => {
    setupGlobalMocks();

    // Set test timeouts
    jest.setTimeout(TEST_CONFIG.timeout);

    // Setup test lifecycle hooks
    beforeAll(async () => {
        await setupTestDatabase();
    });

    afterAll(async () => {
        await cleanupTestDatabase();
    });

    beforeEach(() => {
        jest.clearAllMocks();
    });
};
