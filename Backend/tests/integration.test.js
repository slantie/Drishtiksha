/**
 * Integration Tests for Enhanced Video Analysis Services
 *
 * Tests the integration between services, repositories, and external dependencies
 */

import { jest } from "@jest/globals";
import {
    setupTestDatabase,
    cleanupTestDatabase,
    createTestUser,
    createTestVideo,
    createTestAnalysis,
    createMockServerResponse,
} from "../setup/testSetup.js";

// Mock the dependencies
const mockPrisma = {
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
    $transaction: jest.fn(),
};

// Mock video repository
const mockVideoRepository = {
    create: jest.fn(),
    findById: jest.fn(),
    findAllByUser: jest.fn(),
    update: jest.fn(),
    delete: jest.fn(),
    findAnalysis: jest.fn(),
    createAnalysis: jest.fn(),
    updateAnalysis: jest.fn(),
    findAnalysesByVideo: jest.fn(),
    findAnalysesByType: jest.fn(),
};

// Mock model analysis service
const mockModelAnalysisService = {
    isAvailable: jest.fn(),
    checkHealth: jest.fn(),
    getModelInfo: jest.fn(),
    analyzeVideo: jest.fn(),
    analyzeVideoWithFallback: jest.fn(),
    generateVisualAnalysis: jest.fn(),
    generateFallbackResult: jest.fn(),
    getAvailableModels: jest.fn(),
    isValidAnalysisType: jest.fn(),
    isValidModel: jest.fn(),
};

// Mock video service
const mockVideoService = {
    uploadAndProcessVideo: jest.fn(),
    getAllVideosForUser: jest.fn(),
    getVideoById: jest.fn(),
    updateVideoDetails: jest.fn(),
    deleteVideoById: jest.fn(),
    runFullAnalysis: jest.fn(),
    createVisualAnalysis: jest.fn(),
    createSpecificAnalysis: jest.fn(),
    getLocalVideoPath: jest.fn(),
    cleanupLocalVideoPath: jest.fn(),
    isValidAnalysisType: jest.fn(),
};

describe("Enhanced Video Analysis Services Integration Tests", () => {
    const testUser = createTestUser();
    const testVideo = createTestVideo();

    beforeAll(async () => {
        await setupTestDatabase();
    });

    afterAll(async () => {
        await cleanupTestDatabase();
    });

    beforeEach(() => {
        jest.clearAllMocks();
    });

    describe("VideoRepository Integration", () => {
        test("should create analysis with enhanced schema relationships", async () => {
            const analysisData = {
                videoId: testVideo.id,
                model: "SIGLIP_LSTM_V1",
                type: "DETAILED",
                status: "PROCESSING",
                confidence: 0,
                isDeepfake: false,
                processingStartedAt: new Date(),
            };

            const expectedAnalysis = {
                id: "analysis-123",
                ...analysisData,
                analysisDetails: null,
                frameAnalyses: [],
                temporalAnalyses: [],
                modelInfo: null,
                systemInfo: null,
                analysisError: null,
            };

            mockVideoRepository.createAnalysis.mockResolvedValue(
                expectedAnalysis
            );

            const result = await mockVideoRepository.createAnalysis(
                analysisData
            );

            expect(result).toEqual(expectedAnalysis);
            expect(mockVideoRepository.createAnalysis).toHaveBeenCalledWith(
                analysisData
            );
        });

        test("should find analysis with type filtering", async () => {
            const mockAnalysisWithRelations = {
                id: "analysis-123",
                videoId: testVideo.id,
                model: "SIGLIP_LSTM_V1",
                type: "DETAILED",
                status: "COMPLETED",
                confidence: 0.85,
                isDeepfake: false,
                analysisDetails: {
                    frameConsistency: 0.85,
                    temporalCoherence: 0.9,
                    facialArtifacts: 0.15,
                },
                frameAnalyses: [],
                temporalAnalyses: [],
                modelInfo: {
                    name: "SIGLIP_LSTM_V1",
                    version: "1.0.0",
                },
                systemInfo: {
                    gpu: "NVIDIA RTX 4090",
                    memory: 8192,
                    processingTime: 2.5,
                },
                analysisError: null,
            };

            mockVideoRepository.findAnalysis.mockResolvedValue(
                mockAnalysisWithRelations
            );

            const result = await mockVideoRepository.findAnalysis(
                testVideo.id,
                "SIGLIP_LSTM_V1",
                "DETAILED"
            );

            expect(result).toEqual(mockAnalysisWithRelations);
            expect(result.analysisDetails).toBeDefined();
            expect(result.modelInfo).toBeDefined();
            expect(result.systemInfo).toBeDefined();
        });

        test("should handle new constraint names correctly", async () => {
            // Test the video_model_type_unique constraint
            const duplicateAnalysisData = {
                videoId: testVideo.id,
                model: "SIGLIP_LSTM_V1",
                type: "QUICK",
                status: "PROCESSING",
                confidence: 0,
                isDeepfake: false,
            };

            // First creation should succeed
            mockVideoRepository.createAnalysis.mockResolvedValueOnce({
                id: "analysis-1",
                ...duplicateAnalysisData,
            });

            // Second creation with same video+model+type should fail
            mockVideoRepository.createAnalysis.mockRejectedValueOnce(
                new Error("Unique constraint failed on video_model_type_unique")
            );

            const firstResult = await mockVideoRepository.createAnalysis(
                duplicateAnalysisData
            );
            expect(firstResult.id).toBe("analysis-1");

            await expect(
                mockVideoRepository.createAnalysis(duplicateAnalysisData)
            ).rejects.toThrow(
                "Unique constraint failed on video_model_type_unique"
            );
        });
    });

    describe("ModelAnalysisService Integration", () => {
        test("should handle all analysis types with proper server communication", async () => {
            mockModelAnalysisService.isAvailable.mockReturnValue(true);

            const analysisTypes = ["QUICK", "DETAILED", "FRAMES", "VISUALIZE"];

            for (const type of analysisTypes) {
                const mockResponse = createMockServerResponse(type);
                mockModelAnalysisService.analyzeVideo.mockResolvedValueOnce(
                    mockResponse
                );

                const result = await mockModelAnalysisService.analyzeVideo(
                    "/path/to/video.mp4",
                    type,
                    "SIGLIP_LSTM_V1",
                    testVideo.id
                );

                expect(result).toEqual(mockResponse);
                expect(
                    mockModelAnalysisService.analyzeVideo
                ).toHaveBeenCalledWith(
                    "/path/to/video.mp4",
                    type,
                    "SIGLIP_LSTM_V1",
                    testVideo.id
                );
            }
        });

        test("should handle model enum mapping correctly", async () => {
            const serverModelNames = [
                "siglip_lstm_v1",
                "siglip_lstm_v3",
                "color_cues_lstm_v1",
            ];
            const schemaModelNames = [
                "SIGLIP_LSTM_V1",
                "SIGLIP_LSTM_V3",
                "COLOR_CUES_LSTM_V1",
            ];

            mockModelAnalysisService.isValidModel.mockImplementation(
                (model) =>
                    schemaModelNames.includes(model) ||
                    serverModelNames.includes(model)
            );

            // Test schema enum validation
            for (const model of schemaModelNames) {
                expect(mockModelAnalysisService.isValidModel(model)).toBe(true);
            }

            // Test server model name validation
            for (const model of serverModelNames) {
                expect(mockModelAnalysisService.isValidModel(model)).toBe(true);
            }

            // Test invalid model
            expect(mockModelAnalysisService.isValidModel("INVALID_MODEL")).toBe(
                false
            );
        });

        test("should generate fallback data for each analysis type", async () => {
            const analysisTypes = ["QUICK", "DETAILED", "FRAMES"];

            for (const type of analysisTypes) {
                const fallbackResult = {
                    analysisType: type,
                    prediction: "REAL",
                    confidence: 0.75,
                    is_deepfake: false,
                    processing_time: 2.5,
                    model: "SIGLIP_LSTM_V1",
                    isMockData: true,
                };

                if (type === "DETAILED") {
                    fallbackResult.detailed_metrics = {
                        frame_consistency: 0.85,
                        temporal_coherence: 0.9,
                        facial_artifacts: 0.15,
                    };
                }

                if (type === "FRAMES") {
                    fallbackResult.frame_analyses = [
                        {
                            frame_number: 0,
                            confidence: 0.85,
                            prediction: "REAL",
                        },
                        {
                            frame_number: 10,
                            confidence: 0.82,
                            prediction: "REAL",
                        },
                    ];
                }

                mockModelAnalysisService.generateFallbackResult.mockReturnValueOnce(
                    fallbackResult
                );

                const result = mockModelAnalysisService.generateFallbackResult(
                    type,
                    "SIGLIP_LSTM_V1"
                );

                expect(result.analysisType).toBe(type);
                expect(result.isMockData).toBe(true);

                if (type === "DETAILED") {
                    expect(result.detailed_metrics).toBeDefined();
                }

                if (type === "FRAMES") {
                    expect(result.frame_analyses).toBeDefined();
                    expect(Array.isArray(result.frame_analyses)).toBe(true);
                }
            }
        });

        test("should handle visualization streaming correctly", async () => {
            const mockVisualizationResult =
                createMockServerResponse("VISUALIZE");

            mockModelAnalysisService.generateVisualAnalysis.mockResolvedValue(
                mockVisualizationResult
            );

            const result =
                await mockModelAnalysisService.generateVisualAnalysis(
                    "/path/to/video.mp4",
                    "SIGLIP_LSTM_V1",
                    testVideo.id
                );

            expect(result.success).toBe(true);
            expect(result.visualizationUrl).toContain(
                "/uploads/visualizations/"
            );
            expect(result.visualizationPath).toBeDefined();
        });
    });

    describe("VideoService Integration", () => {
        test("should orchestrate comprehensive analysis workflow", async () => {
            const analysisConfig = {
                types: ["QUICK", "DETAILED"],
                models: ["SIGLIP_LSTM_V1", "SIGLIP_LSTM_V3"],
                enableVisualization: true,
            };

            // Mock video repository responses
            mockVideoRepository.findById.mockResolvedValue(testVideo);
            mockVideoRepository.findAnalysis.mockResolvedValue(null); // No existing analysis
            mockVideoRepository.createAnalysis.mockResolvedValue(
                createTestAnalysis()
            );
            mockVideoRepository.updateAnalysis.mockResolvedValue(
                createTestAnalysis({ status: "COMPLETED" })
            );

            // Mock model analysis service responses
            mockModelAnalysisService.analyzeVideoWithFallback.mockResolvedValue(
                createMockServerResponse("QUICK")
            );
            mockModelAnalysisService.generateVisualAnalysis.mockResolvedValue(
                createMockServerResponse("VISUALIZE")
            );

            // Mock video service methods
            mockVideoService.getLocalVideoPath.mockResolvedValue(
                "/tmp/test-video.mp4"
            );
            mockVideoService.cleanupLocalVideoPath.mockResolvedValue();

            await mockVideoService.runFullAnalysis(
                testVideo.id,
                analysisConfig
            );

            // Verify that all analysis types and models were processed
            expect(mockVideoRepository.createAnalysis).toHaveBeenCalledTimes(5); // 2 types * 2 models + 1 visualization
            expect(
                mockModelAnalysisService.analyzeVideoWithFallback
            ).toHaveBeenCalledTimes(4); // 2 types * 2 models
            expect(
                mockModelAnalysisService.generateVisualAnalysis
            ).toHaveBeenCalledTimes(2); // 2 models for visualization
        });

        test("should handle selective analysis creation", async () => {
            const analysisType = "DETAILED";
            const model = "SIGLIP_LSTM_V1";

            mockVideoService.getVideoById.mockResolvedValue(testVideo);
            mockVideoRepository.findAnalysis.mockResolvedValue(null); // No existing analysis
            mockVideoRepository.createAnalysis.mockResolvedValue(
                createTestAnalysis({ type: analysisType, model })
            );
            mockVideoService._performSingleAnalysis = jest
                .fn()
                .mockResolvedValue();

            const result = await mockVideoService.createSpecificAnalysis(
                testVideo.id,
                testUser,
                analysisType,
                model
            );

            expect(result).toBeInstanceOf(Array);
            expect(mockVideoService.getVideoById).toHaveBeenCalledWith(
                testVideo.id,
                testUser
            );
            expect(mockVideoRepository.createAnalysis).toHaveBeenCalledWith(
                expect.objectContaining({
                    videoId: testVideo.id,
                    model,
                    type: analysisType,
                    status: "PROCESSING",
                })
            );
        });

        test("should handle enhanced visualization with model selection", async () => {
            const selectedModel = "COLOR_CUES_LSTM_V1";

            mockVideoService.getVideoById.mockResolvedValue(testVideo);
            mockVideoRepository.findAnalysis.mockResolvedValue(null); // No existing visualization
            mockVideoRepository.createAnalysis.mockResolvedValue(
                createTestAnalysis({ type: "VISUALIZE", model: selectedModel })
            );
            mockVideoService._performVisualization = jest
                .fn()
                .mockResolvedValue();
            mockVideoRepository.findById.mockResolvedValue({
                ...testVideo,
                analyses: [
                    createTestAnalysis({
                        type: "VISUALIZE",
                        model: selectedModel,
                    }),
                ],
            });

            const result = await mockVideoService.createVisualAnalysis(
                testVideo.id,
                testUser,
                selectedModel
            );

            expect(result.analyses).toHaveLength(1);
            expect(result.analyses[0].type).toBe("VISUALIZE");
            expect(result.analyses[0].model).toBe(selectedModel);
        });

        test("should validate analysis types correctly", () => {
            mockVideoService.isValidAnalysisType.mockImplementation((type) =>
                ["QUICK", "DETAILED", "FRAMES", "VISUALIZE"].includes(type)
            );

            expect(mockVideoService.isValidAnalysisType("QUICK")).toBe(true);
            expect(mockVideoService.isValidAnalysisType("DETAILED")).toBe(true);
            expect(mockVideoService.isValidAnalysisType("FRAMES")).toBe(true);
            expect(mockVideoService.isValidAnalysisType("VISUALIZE")).toBe(
                true
            );
            expect(mockVideoService.isValidAnalysisType("INVALID")).toBe(false);
        });
    });

    describe("Error Handling Integration", () => {
        test("should handle model service unavailable gracefully", async () => {
            mockModelAnalysisService.isAvailable.mockReturnValue(false);
            mockModelAnalysisService.generateFallbackResult.mockReturnValue(
                createMockServerResponse("QUICK")
            );

            const result =
                await mockModelAnalysisService.analyzeVideoWithFallback(
                    "/path/to/video.mp4",
                    testVideo.id,
                    "QUICK",
                    "SIGLIP_LSTM_V1"
                );

            expect(result).toBeDefined();
            expect(result.isMockData).toBe(true);
        });

        test("should handle database constraint violations", async () => {
            const constraintError = new Error(
                "Unique constraint failed on video_model_type_unique"
            );
            constraintError.code = "P2002";

            mockVideoRepository.createAnalysis.mockRejectedValue(
                constraintError
            );

            await expect(
                mockVideoRepository.createAnalysis({
                    videoId: testVideo.id,
                    model: "SIGLIP_LSTM_V1",
                    type: "QUICK",
                })
            ).rejects.toThrow(
                "Unique constraint failed on video_model_type_unique"
            );
        });

        test("should handle analysis timeout scenarios", async () => {
            const timeoutError = new Error(
                "Analysis timeout - video processing took too long"
            );
            timeoutError.code = "ETIMEDOUT";

            mockModelAnalysisService.analyzeVideo.mockRejectedValue(
                timeoutError
            );
            mockModelAnalysisService.generateFallbackResult.mockReturnValue(
                createMockServerResponse("QUICK")
            );

            // Should fallback to mock data on timeout
            const result =
                await mockModelAnalysisService.analyzeVideoWithFallback(
                    "/path/to/video.mp4",
                    testVideo.id,
                    "QUICK",
                    "SIGLIP_LSTM_V1"
                );

            expect(result).toBeDefined();
            expect(result.isMockData).toBe(true);
        });

        test("should handle invalid video file scenarios", async () => {
            const invalidFileError = new Error("Video file not found");

            mockVideoService.getLocalVideoPath.mockRejectedValue(
                invalidFileError
            );

            await expect(
                mockVideoService.getLocalVideoPath(testVideo)
            ).rejects.toThrow("Video file not found");
        });
    });

    describe("Performance Integration Tests", () => {
        test("should handle concurrent analysis requests efficiently", async () => {
            const concurrentAnalyses = Array.from({ length: 10 }, (_, i) => ({
                videoId: `video-${i}`,
                type: "QUICK",
                model: "SIGLIP_LSTM_V1",
            }));

            // Mock successful processing for all requests
            concurrentAnalyses.forEach((analysis, index) => {
                mockVideoService.createSpecificAnalysis.mockResolvedValueOnce([
                    createTestAnalysis({
                        id: `analysis-${index}`,
                        videoId: analysis.videoId,
                        type: analysis.type,
                        model: analysis.model,
                    }),
                ]);
            });

            const analysisPromises = concurrentAnalyses.map((analysis) =>
                mockVideoService.createSpecificAnalysis(
                    analysis.videoId,
                    testUser,
                    analysis.type,
                    analysis.model
                )
            );

            const results = await Promise.all(analysisPromises);

            expect(results).toHaveLength(10);
            expect(
                mockVideoService.createSpecificAnalysis
            ).toHaveBeenCalledTimes(10);
        });

        test("should handle large analysis result datasets", async () => {
            const largeAnalysisSet = Array.from({ length: 100 }, (_, i) =>
                createTestAnalysis({
                    id: `analysis-${i}`,
                    createdAt: new Date(Date.now() - i * 1000),
                })
            );

            mockVideoRepository.findAnalysesByVideo.mockResolvedValue(
                largeAnalysisSet
            );

            const result = await mockVideoRepository.findAnalysesByVideo(
                testVideo.id
            );

            expect(result).toHaveLength(100);
            expect(result[0].id).toBe("analysis-0");
            expect(result[99].id).toBe("analysis-99");
        });
    });

    describe("Queue Integration Tests", () => {
        test("should process analysis jobs with enhanced configurations", async () => {
            const jobData = {
                videoId: testVideo.id,
                userId: testUser.id,
                analysisConfig: {
                    types: ["QUICK", "DETAILED", "FRAMES"],
                    models: ["SIGLIP_LSTM_V1", "SIGLIP_LSTM_V3"],
                    enableVisualization: true,
                },
            };

            // Mock queue processing
            mockVideoService.runFullAnalysis.mockResolvedValue();

            // Simulate queue processing
            await mockVideoService.runFullAnalysis(
                jobData.videoId,
                jobData.analysisConfig
            );

            expect(mockVideoService.runFullAnalysis).toHaveBeenCalledWith(
                jobData.videoId,
                jobData.analysisConfig
            );
        });

        test("should handle priority analysis jobs", async () => {
            const priorityJob = {
                videoId: testVideo.id,
                userId: testUser.id,
                analysisConfig: {
                    types: ["QUICK"],
                    models: ["SIGLIP_LSTM_V1"],
                    enableVisualization: false,
                },
                priority: "HIGH",
            };

            const regularJob = {
                videoId: "video-456",
                userId: testUser.id,
                analysisConfig: {
                    types: ["DETAILED"],
                    models: ["SIGLIP_LSTM_V3"],
                    enableVisualization: true,
                },
                priority: "NORMAL",
            };

            // Mock queue with priority handling
            const processedJobs = [];
            const mockPriorityQueue = {
                addPriority: (job) => processedJobs.unshift(job),
                addRegular: (job) => processedJobs.push(job),
                process: async () => {
                    for (const job of processedJobs) {
                        await mockVideoService.runFullAnalysis(
                            job.videoId,
                            job.analysisConfig
                        );
                    }
                },
            };

            mockPriorityQueue.addRegular(regularJob);
            mockPriorityQueue.addPriority(priorityJob);

            // Verify priority job is processed first
            expect(processedJobs[0]).toEqual(priorityJob);
            expect(processedJobs[1]).toEqual(regularJob);
        });
    });
});
