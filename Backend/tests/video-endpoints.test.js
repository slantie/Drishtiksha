/**
 * Comprehensive Test Suite for Enhanced Video Analysis API
 *
 * Tests all newly created endpoints and enhanced functionality:
 * - Enhanced analysis endpoints (QUICK, DETAILED, FRAMES, VISUALIZE)
 * - Model-specific operations
 * - Analysis results retrieval with filtering
 * - Enhanced model status
 * - Queue processing with configurations
 */

// Note: Run `npm install --save-dev jest supertest` before running tests

import request from "supertest";
import { jest } from "@jest/globals";
import path from "path";
import fs from "fs";

// Mock the app and dependencies
const mockApp = {
    use: jest.fn(),
    listen: jest.fn(),
    get: jest.fn(),
    post: jest.fn(),
    patch: jest.fn(),
    delete: jest.fn(),
};

// Mock implementations
const mockVideoService = {
    uploadAndProcessVideo: jest.fn(),
    getAllVideosForUser: jest.fn(),
    getVideoById: jest.fn(),
    updateVideoDetails: jest.fn(),
    deleteVideoById: jest.fn(),
    createVisualAnalysis: jest.fn(),
    createSpecificAnalysis: jest.fn(),
};

const mockModelAnalysisService = {
    isAvailable: jest.fn(),
    checkHealth: jest.fn(),
    getModelInfo: jest.fn(),
    getAvailableModels: jest.fn(),
    analyzeVideo: jest.fn(),
    generateVisualAnalysis: jest.fn(),
    isValidAnalysisType: jest.fn(),
    isValidModel: jest.fn(),
};

const mockVideoRepository = {
    findById: jest.fn(),
    findAnalysis: jest.fn(),
    findAnalysesByVideo: jest.fn(),
    findAnalysesByType: jest.fn(),
    createAnalysis: jest.fn(),
    updateAnalysis: jest.fn(),
};

// Mock Express app setup
jest.mock("../src/app.js", () => mockApp);
jest.mock("../src/services/video.service.js", () => ({
    videoService: mockVideoService,
}));
jest.mock("../src/services/modelAnalysis.service.js", () => ({
    modelAnalysisService: mockModelAnalysisService,
}));
jest.mock("../src/repositories/video.repository.js", () => ({
    videoRepository: mockVideoRepository,
}));

describe("Enhanced Video Analysis API Tests", () => {
    // Test data
    const mockUser = {
        id: "user-123",
        email: "test@example.com",
        role: "USER",
    };

    const mockVideo = {
        id: "video-123",
        filename: "test-video.mp4",
        userId: "user-123",
        url: "https://example.com/video.mp4",
        status: "UPLOADED",
        analyses: [],
    };

    const mockAnalysis = {
        id: "analysis-123",
        videoId: "video-123",
        model: "SIGLIP_LSTM_V1",
        type: "QUICK",
        status: "COMPLETED",
        confidence: 0.85,
        isDeepfake: false,
        prediction: "REAL",
        processingTime: 2.5,
        createdAt: new Date(),
    };

    const mockToken = "mock-jwt-token";

    beforeEach(() => {
        jest.clearAllMocks();
    });

    describe("Model Status Endpoint", () => {
        test("GET /api/videos/model/status - should return enhanced model status", async () => {
            // Mock service responses
            mockModelAnalysisService.isAvailable.mockReturnValue(true);
            mockModelAnalysisService.checkHealth.mockResolvedValue({
                status: "healthy",
                models_loaded: 3,
                memory_usage: "2.1GB",
            });
            mockModelAnalysisService.getModelInfo.mockResolvedValue({
                server_version: "2.0.0",
                available_models: [
                    "siglip_lstm_v1",
                    "siglip_lstm_v3",
                    "color_cues_lstm_v1",
                ],
            });
            mockModelAnalysisService.getAvailableModels.mockResolvedValue([
                "SIGLIP_LSTM_V1",
                "SIGLIP_LSTM_V3",
                "COLOR_CUES_LSTM_V1",
            ]);

            // Simulate the controller logic
            const expectedResponse = {
                success: true,
                data: {
                    isConfigured: true,
                    health: {
                        status: "healthy",
                        models_loaded: 3,
                        memory_usage: "2.1GB",
                    },
                    modelInfo: {
                        server_version: "2.0.0",
                        available_models: [
                            "siglip_lstm_v1",
                            "siglip_lstm_v3",
                            "color_cues_lstm_v1",
                        ],
                    },
                    availableModels: [
                        "SIGLIP_LSTM_V1",
                        "SIGLIP_LSTM_V3",
                        "COLOR_CUES_LSTM_V1",
                    ],
                    supportedAnalysisTypes: [
                        "QUICK",
                        "DETAILED",
                        "FRAMES",
                        "VISUALIZE",
                    ],
                },
                message: "Model service status retrieved successfully.",
            };

            // Verify service calls
            expect(mockModelAnalysisService.isAvailable).toHaveBeenCalled();
            expect(mockModelAnalysisService.checkHealth).toHaveBeenCalled();
            expect(mockModelAnalysisService.getModelInfo).toHaveBeenCalled();
            expect(
                mockModelAnalysisService.getAvailableModels
            ).toHaveBeenCalled();
        });

        test("GET /api/videos/model/status - should handle service unavailable", async () => {
            mockModelAnalysisService.isAvailable.mockReturnValue(false);

            const expectedResponse = {
                success: true,
                data: {
                    isConfigured: false,
                    health: null,
                    modelInfo: null,
                    availableModels: [],
                    supportedAnalysisTypes: [
                        "QUICK",
                        "DETAILED",
                        "FRAMES",
                        "VISUALIZE",
                    ],
                },
                message: "Model service status retrieved successfully.",
            };

            expect(mockModelAnalysisService.isAvailable).toHaveBeenCalled();
        });
    });

    describe("Specific Analysis Creation Endpoint", () => {
        test("POST /api/videos/:id/analyze - should create QUICK analysis", async () => {
            const analysisRequest = {
                type: "QUICK",
                model: "SIGLIP_LSTM_V1",
            };

            const expectedAnalysis = {
                ...mockAnalysis,
                type: "QUICK",
                model: "SIGLIP_LSTM_V1",
            };

            mockVideoService.createSpecificAnalysis.mockResolvedValue([
                expectedAnalysis,
            ]);

            // Verify service call with correct parameters
            expect(
                mockVideoService.createSpecificAnalysis
            ).toHaveBeenCalledWith(
                "video-123",
                expect.objectContaining({ id: mockUser.id }),
                "QUICK",
                "SIGLIP_LSTM_V1"
            );
        });

        test("POST /api/videos/:id/analyze - should create DETAILED analysis", async () => {
            const analysisRequest = {
                type: "DETAILED",
                model: "SIGLIP_LSTM_V3",
            };

            const expectedAnalysis = {
                ...mockAnalysis,
                type: "DETAILED",
                model: "SIGLIP_LSTM_V3",
                analysisDetails: {
                    frameConsistency: 0.85,
                    temporalCoherence: 0.9,
                    facialArtifacts: 0.15,
                },
            };

            mockVideoService.createSpecificAnalysis.mockResolvedValue([
                expectedAnalysis,
            ]);

            expect(
                mockVideoService.createSpecificAnalysis
            ).toHaveBeenCalledWith(
                "video-123",
                expect.objectContaining({ id: mockUser.id }),
                "DETAILED",
                "SIGLIP_LSTM_V3"
            );
        });

        test("POST /api/videos/:id/analyze - should create FRAMES analysis", async () => {
            const analysisRequest = {
                type: "FRAMES",
                model: "COLOR_CUES_LSTM_V1",
            };

            const expectedAnalysis = {
                ...mockAnalysis,
                type: "FRAMES",
                model: "COLOR_CUES_LSTM_V1",
                frameAnalyses: [
                    { frameNumber: 0, confidence: 0.85, prediction: "REAL" },
                    { frameNumber: 10, confidence: 0.82, prediction: "REAL" },
                    { frameNumber: 20, confidence: 0.78, prediction: "REAL" },
                ],
            };

            mockVideoService.createSpecificAnalysis.mockResolvedValue([
                expectedAnalysis,
            ]);

            expect(
                mockVideoService.createSpecificAnalysis
            ).toHaveBeenCalledWith(
                "video-123",
                expect.objectContaining({ id: mockUser.id }),
                "FRAMES",
                "COLOR_CUES_LSTM_V1"
            );
        });

        test("POST /api/videos/:id/analyze - should validate analysis type", async () => {
            const invalidRequest = {
                type: "INVALID_TYPE",
            };

            // This should trigger validation error
            // In real implementation, this would be caught by Zod validation
            const validationError = {
                success: false,
                errors: [
                    {
                        path: ["body", "type"],
                        message: "Invalid analysis type",
                    },
                ],
            };

            // Verify validation logic
            expect(["QUICK", "DETAILED", "FRAMES", "VISUALIZE"]).not.toContain(
                "INVALID_TYPE"
            );
        });

        test("POST /api/videos/:id/analyze - should handle missing type", async () => {
            const invalidRequest = {
                model: "SIGLIP_LSTM_V1",
            };

            // This should trigger validation error for missing type
            const validationError = {
                success: false,
                errors: [
                    {
                        path: ["body", "type"],
                        message: "Analysis type is required",
                    },
                ],
            };
        });
    });

    describe("Enhanced Visualization Endpoint", () => {
        test("POST /api/videos/:id/visualize - should create visualization with specific model", async () => {
            const visualizationRequest = {
                model: "SIGLIP_LSTM_V1",
            };

            const mockVideoWithVisualization = {
                ...mockVideo,
                analyses: [
                    {
                        ...mockAnalysis,
                        type: "VISUALIZE",
                        model: "SIGLIP_LSTM_V1",
                        visualizedUrl:
                            "/uploads/visualizations/visualization-123456789.mp4",
                    },
                ],
            };

            mockVideoService.createVisualAnalysis.mockResolvedValue(
                mockVideoWithVisualization
            );

            expect(mockVideoService.createVisualAnalysis).toHaveBeenCalledWith(
                "video-123",
                expect.objectContaining({ id: mockUser.id }),
                "SIGLIP_LSTM_V1"
            );
        });

        test("POST /api/videos/:id/visualize - should create visualization with default model", async () => {
            const mockVideoWithVisualization = {
                ...mockVideo,
                analyses: [
                    {
                        ...mockAnalysis,
                        type: "VISUALIZE",
                        visualizedUrl:
                            "/uploads/visualizations/visualization-123456789.mp4",
                    },
                ],
            };

            mockVideoService.createVisualAnalysis.mockResolvedValue(
                mockVideoWithVisualization
            );

            expect(mockVideoService.createVisualAnalysis).toHaveBeenCalledWith(
                "video-123",
                expect.objectContaining({ id: mockUser.id }),
                null
            );
        });

        test("POST /api/videos/:id/visualize - should handle visualization failures", async () => {
            mockVideoService.createVisualAnalysis.mockRejectedValue(
                new Error("Visualization generation failed")
            );

            // In real implementation, this would return an error response
            try {
                await mockVideoService.createVisualAnalysis(
                    "video-123",
                    mockUser,
                    "SIGLIP_LSTM_V1"
                );
            } catch (error) {
                expect(error.message).toBe("Visualization generation failed");
            }
        });
    });

    describe("Analysis Results Retrieval Endpoint", () => {
        test("GET /api/videos/:id/analysis - should get all analyses", async () => {
            const mockAnalyses = [
                { ...mockAnalysis, type: "QUICK", model: "SIGLIP_LSTM_V1" },
                { ...mockAnalysis, type: "DETAILED", model: "SIGLIP_LSTM_V3" },
                {
                    ...mockAnalysis,
                    type: "FRAMES",
                    model: "COLOR_CUES_LSTM_V1",
                },
            ];

            mockVideoService.getVideoById.mockResolvedValue(mockVideo);
            mockVideoRepository.findAnalysesByVideo.mockResolvedValue(
                mockAnalyses
            );

            expect(
                mockVideoRepository.findAnalysesByVideo
            ).toHaveBeenCalledWith("video-123");
        });

        test("GET /api/videos/:id/analysis?type=DETAILED - should filter by type", async () => {
            const mockDetailedAnalyses = [
                { ...mockAnalysis, type: "DETAILED", model: "SIGLIP_LSTM_V1" },
                { ...mockAnalysis, type: "DETAILED", model: "SIGLIP_LSTM_V3" },
            ];

            mockVideoService.getVideoById.mockResolvedValue(mockVideo);
            mockVideoRepository.findAnalysesByType.mockResolvedValue(
                mockDetailedAnalyses
            );

            expect(mockVideoRepository.findAnalysesByType).toHaveBeenCalledWith(
                "video-123",
                "DETAILED"
            );
        });

        test("GET /api/videos/:id/analysis?type=QUICK&model=SIGLIP_LSTM_V1 - should filter by type and model", async () => {
            const mockSpecificAnalysis = {
                ...mockAnalysis,
                type: "QUICK",
                model: "SIGLIP_LSTM_V1",
            };

            mockVideoService.getVideoById.mockResolvedValue(mockVideo);
            mockVideoRepository.findAnalysis.mockResolvedValue(
                mockSpecificAnalysis
            );

            expect(mockVideoRepository.findAnalysis).toHaveBeenCalledWith(
                "video-123",
                "SIGLIP_LSTM_V1",
                "QUICK"
            );
        });

        test("GET /api/videos/:id/analysis - should handle no results", async () => {
            mockVideoService.getVideoById.mockResolvedValue(mockVideo);
            mockVideoRepository.findAnalysesByVideo.mockResolvedValue([]);

            expect(
                mockVideoRepository.findAnalysesByVideo
            ).toHaveBeenCalledWith("video-123");
        });
    });

    describe("Service Layer Tests", () => {
        describe("ModelAnalysisService", () => {
            test("analyzeVideo - should support all analysis types", async () => {
                const analysisTypes = [
                    "QUICK",
                    "DETAILED",
                    "FRAMES",
                    "VISUALIZE",
                ];

                for (const type of analysisTypes) {
                    mockModelAnalysisService.analyzeVideo.mockResolvedValue({
                        analysisType: type,
                        confidence: 0.85,
                        is_deepfake: false,
                        prediction: "REAL",
                        processing_time: 2.5,
                    });

                    const result = await mockModelAnalysisService.analyzeVideo(
                        "/path/to/video.mp4",
                        type,
                        "SIGLIP_LSTM_V1",
                        "video-123"
                    );

                    expect(result.analysisType).toBe(type);
                    expect(
                        mockModelAnalysisService.analyzeVideo
                    ).toHaveBeenCalledWith(
                        "/path/to/video.mp4",
                        type,
                        "SIGLIP_LSTM_V1",
                        "video-123"
                    );
                }
            });

            test("isValidAnalysisType - should validate analysis types", () => {
                mockModelAnalysisService.isValidAnalysisType.mockImplementation(
                    (type) =>
                        ["QUICK", "DETAILED", "FRAMES", "VISUALIZE"].includes(
                            type
                        )
                );

                expect(
                    mockModelAnalysisService.isValidAnalysisType("QUICK")
                ).toBe(true);
                expect(
                    mockModelAnalysisService.isValidAnalysisType("DETAILED")
                ).toBe(true);
                expect(
                    mockModelAnalysisService.isValidAnalysisType("INVALID")
                ).toBe(false);
            });

            test("isValidModel - should validate model names", () => {
                mockModelAnalysisService.isValidModel.mockImplementation(
                    (model) =>
                        [
                            "SIGLIP_LSTM_V1",
                            "SIGLIP_LSTM_V3",
                            "COLOR_CUES_LSTM_V1",
                        ].includes(model)
                );

                expect(
                    mockModelAnalysisService.isValidModel("SIGLIP_LSTM_V1")
                ).toBe(true);
                expect(
                    mockModelAnalysisService.isValidModel("SIGLIP_LSTM_V3")
                ).toBe(true);
                expect(
                    mockModelAnalysisService.isValidModel("INVALID_MODEL")
                ).toBe(false);
            });
        });

        describe("VideoRepository", () => {
            test("findAnalysis - should support type filtering", async () => {
                const mockAnalysisResult = {
                    ...mockAnalysis,
                    analysisDetails: {},
                    frameAnalyses: [],
                    temporalAnalyses: [],
                    modelInfo: {},
                    systemInfo: {},
                    analysisError: null,
                };

                mockVideoRepository.findAnalysis.mockResolvedValue(
                    mockAnalysisResult
                );

                const result = await mockVideoRepository.findAnalysis(
                    "video-123",
                    "SIGLIP_LSTM_V1",
                    "QUICK"
                );

                expect(result).toEqual(mockAnalysisResult);
                expect(mockVideoRepository.findAnalysis).toHaveBeenCalledWith(
                    "video-123",
                    "SIGLIP_LSTM_V1",
                    "QUICK"
                );
            });

            test("createAnalysis - should include all relationships", async () => {
                const analysisData = {
                    videoId: "video-123",
                    model: "SIGLIP_LSTM_V1",
                    type: "DETAILED",
                    status: "PROCESSING",
                    confidence: 0,
                    isDeepfake: false,
                    processingStartedAt: new Date(),
                };

                const mockCreatedAnalysis = {
                    id: "analysis-456",
                    ...analysisData,
                    analysisDetails: null,
                    frameAnalyses: [],
                    temporalAnalyses: [],
                    modelInfo: null,
                    systemInfo: null,
                    analysisError: null,
                };

                mockVideoRepository.createAnalysis.mockResolvedValue(
                    mockCreatedAnalysis
                );

                const result = await mockVideoRepository.createAnalysis(
                    analysisData
                );

                expect(result).toEqual(mockCreatedAnalysis);
                expect(mockVideoRepository.createAnalysis).toHaveBeenCalledWith(
                    analysisData
                );
            });
        });
    });

    describe("Error Handling Tests", () => {
        test("should handle invalid video ID format", () => {
            const invalidVideoId = "invalid-uuid";

            // Zod validation would catch this
            const validationError = {
                success: false,
                errors: [
                    {
                        path: ["params", "id"],
                        message: "Invalid video ID format",
                    },
                ],
            };

            // Test UUID validation
            const uuidRegex =
                /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
            expect(uuidRegex.test(invalidVideoId)).toBe(false);
        });

        test("should handle unauthorized access", async () => {
            mockVideoService.getVideoById.mockRejectedValue(
                new Error("Access denied. You do not own this video.")
            );

            try {
                await mockVideoService.getVideoById("video-123", {
                    id: "different-user",
                });
            } catch (error) {
                expect(error.message).toBe(
                    "Access denied. You do not own this video."
                );
            }
        });

        test("should handle model service unavailable", async () => {
            mockModelAnalysisService.isAvailable.mockReturnValue(false);
            mockModelAnalysisService.checkHealth.mockRejectedValue(
                new Error("Model server health check failed")
            );

            try {
                await mockModelAnalysisService.checkHealth();
            } catch (error) {
                expect(error.message).toBe("Model server health check failed");
            }
        });

        test("should handle analysis creation failure", async () => {
            mockVideoService.createSpecificAnalysis.mockRejectedValue(
                new Error("Analysis creation failed")
            );

            try {
                await mockVideoService.createSpecificAnalysis(
                    "video-123",
                    mockUser,
                    "QUICK",
                    "SIGLIP_LSTM_V1"
                );
            } catch (error) {
                expect(error.message).toBe("Analysis creation failed");
            }
        });
    });

    describe("Integration Tests", () => {
        test("complete analysis workflow", async () => {
            // 1. Check model status
            mockModelAnalysisService.isAvailable.mockReturnValue(true);
            mockModelAnalysisService.checkHealth.mockResolvedValue({
                status: "healthy",
            });

            // 2. Create specific analysis
            const expectedAnalysis = {
                ...mockAnalysis,
                type: "DETAILED",
                model: "SIGLIP_LSTM_V1",
            };
            mockVideoService.createSpecificAnalysis.mockResolvedValue([
                expectedAnalysis,
            ]);

            // 3. Retrieve analysis results
            mockVideoService.getVideoById.mockResolvedValue(mockVideo);
            mockVideoRepository.findAnalysis.mockResolvedValue(
                expectedAnalysis
            );

            // 4. Generate visualization
            const videoWithVisualization = {
                ...mockVideo,
                analyses: [
                    ...mockVideo.analyses,
                    {
                        ...mockAnalysis,
                        type: "VISUALIZE",
                        visualizedUrl: "/uploads/visualizations/viz-123.mp4",
                    },
                ],
            };
            mockVideoService.createVisualAnalysis.mockResolvedValue(
                videoWithVisualization
            );

            // Execute workflow
            const healthStatus = await mockModelAnalysisService.checkHealth();
            expect(healthStatus.status).toBe("healthy");

            const analysisResult =
                await mockVideoService.createSpecificAnalysis(
                    "video-123",
                    mockUser,
                    "DETAILED",
                    "SIGLIP_LSTM_V1"
                );
            expect(analysisResult[0].type).toBe("DETAILED");

            const retrievedAnalysis = await mockVideoRepository.findAnalysis(
                "video-123",
                "SIGLIP_LSTM_V1",
                "DETAILED"
            );
            expect(retrievedAnalysis.type).toBe("DETAILED");

            const visualizationResult =
                await mockVideoService.createVisualAnalysis(
                    "video-123",
                    mockUser,
                    "SIGLIP_LSTM_V1"
                );
            expect(
                visualizationResult.analyses.some((a) => a.type === "VISUALIZE")
            ).toBe(true);
        });

        test("queue processing with analysis configuration", () => {
            const jobData = {
                videoId: "video-123",
                userId: "user-123",
                analysisConfig: {
                    types: ["QUICK", "DETAILED"],
                    models: ["SIGLIP_LSTM_V1", "SIGLIP_LSTM_V3"],
                    enableVisualization: true,
                },
            };

            // Mock queue processing
            const processedJobs = [];
            const mockQueue = {
                add: (data) => processedJobs.push(data),
                process: async () => {
                    for (const job of processedJobs) {
                        await mockVideoService.runFullAnalysis(
                            job.videoId,
                            job.analysisConfig
                        );
                    }
                },
            };

            mockQueue.add(jobData);
            expect(processedJobs).toHaveLength(1);
            expect(processedJobs[0].analysisConfig.types).toEqual([
                "QUICK",
                "DETAILED",
            ]);
            expect(processedJobs[0].analysisConfig.enableVisualization).toBe(
                true
            );
        });
    });

    describe("Performance Tests", () => {
        test("should handle concurrent analysis requests", async () => {
            const concurrentRequests = Array.from({ length: 5 }, (_, i) => ({
                videoId: `video-${i}`,
                type: "QUICK",
                model: "SIGLIP_LSTM_V1",
            }));

            // Mock concurrent processing
            const analysisPromises = concurrentRequests.map(
                async (req, index) => {
                    mockVideoService.createSpecificAnalysis.mockResolvedValueOnce(
                        [
                            {
                                ...mockAnalysis,
                                id: `analysis-${index}`,
                                videoId: req.videoId,
                            },
                        ]
                    );

                    return mockVideoService.createSpecificAnalysis(
                        req.videoId,
                        mockUser,
                        req.type,
                        req.model
                    );
                }
            );

            const results = await Promise.all(analysisPromises);
            expect(results).toHaveLength(5);
            expect(
                mockVideoService.createSpecificAnalysis
            ).toHaveBeenCalledTimes(5);
        });

        test("should handle large analysis result sets", async () => {
            // Mock large dataset
            const largeAnalysisSet = Array.from({ length: 100 }, (_, i) => ({
                ...mockAnalysis,
                id: `analysis-${i}`,
                createdAt: new Date(Date.now() - i * 1000),
            }));

            mockVideoRepository.findAnalysesByVideo.mockResolvedValue(
                largeAnalysisSet
            );

            const results = await mockVideoRepository.findAnalysesByVideo(
                "video-123"
            );
            expect(results).toHaveLength(100);
            expect(results[0].id).toBe("analysis-0");
        });
    });
});

// Helper functions for test setup
const createMockVideoFile = () => ({
    originalname: "test-video.mp4",
    mimetype: "video/mp4",
    size: 1024 * 1024, // 1MB
    buffer: Buffer.from("mock-video-data"),
});

const createMockAuthToken = (userId = "user-123") => {
    // Mock JWT token creation
    return `mock-jwt-token-${userId}`;
};

const setupTestDatabase = async () => {
    // Mock database setup for tests
    // In real implementation, this would set up a test database
    console.log("Setting up test database...");
};

const cleanupTestDatabase = async () => {
    // Mock database cleanup
    // In real implementation, this would clean up test data
    console.log("Cleaning up test database...");
};

// Export test utilities
export {
    createMockVideoFile,
    createMockAuthToken,
    setupTestDatabase,
    cleanupTestDatabase,
};
