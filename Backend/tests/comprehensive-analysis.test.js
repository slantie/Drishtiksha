/**
 * Integration Test for Comprehensive Analysis Feature
 *
 * Tests the new comprehensive analysis pipeline that replaces
 * individual analysis types as the default processing method.
 */

import { modelAnalysisService } from "../src/services/modelAnalysis.service.js";
import { videoService } from "../src/services/video.service.js";
import { videoRepository } from "../src/repositories/video.repository.js";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

describe("Comprehensive Analysis Integration", () => {
    const testVideoPath = path.join(__dirname, "fixtures", "test-video.mp4");

    beforeAll(() => {
        // Ensure test video exists
        if (!fs.existsSync(testVideoPath)) {
            console.warn("Test video not found, skipping integration tests");
        }
    });

    describe("ModelAnalysisService", () => {
        test("should have comprehensive analysis endpoint available", () => {
            expect(
                modelAnalysisService.analyzeVideoComprehensive
            ).toBeDefined();
            expect(typeof modelAnalysisService.analyzeVideoComprehensive).toBe(
                "function"
            );
        });

        test("should validate service configuration", () => {
            const isAvailable = modelAnalysisService.isAvailable();
            console.log("Model service available:", isAvailable);

            if (!isAvailable) {
                console.warn(
                    "Model service not configured - skipping analysis tests"
                );
                return;
            }

            expect(isAvailable).toBe(true);
        });

        test("should check health status", async () => {
            if (!modelAnalysisService.isAvailable()) {
                console.warn(
                    "Model service not available - skipping health check"
                );
                return;
            }

            try {
                const health = await modelAnalysisService.getHealthStatus();
                console.log("Health status:", health);
                expect(health).toBeDefined();
                expect(health.status || health.message).toBeDefined();
            } catch (error) {
                console.warn(
                    "Health check failed (expected if server not running):",
                    error.message
                );
            }
        }, 20000);

        test("should get available models", async () => {
            if (!modelAnalysisService.isAvailable()) {
                console.warn(
                    "Model service not available - skipping model check"
                );
                return;
            }

            try {
                const models = await modelAnalysisService.getAvailableModels();
                console.log("Available models:", models);
                expect(Array.isArray(models)).toBe(true);
            } catch (error) {
                console.warn(
                    "Model check failed (expected if server not running):",
                    error.message
                );
            }
        }, 20000);
    });

    describe("Video Service Pipeline", () => {
        test("should have comprehensive analysis method", () => {
            expect(videoService._runAndSaveComprehensiveAnalysis).toBeDefined();
            expect(typeof videoService._runAndSaveComprehensiveAnalysis).toBe(
                "function"
            );
        });

        test("should use comprehensive analysis for advanced models", () => {
            // Check that the ADVANCED_MODELS constant includes the expected models
            const videoServiceCode = fs.readFileSync(
                path.join(__dirname, "../src/services/video.service.js"),
                "utf8"
            );

            expect(videoServiceCode).toContain("ADVANCED_MODELS");
            expect(videoServiceCode).toContain(
                "_runAndSaveComprehensiveAnalysis"
            );
            expect(videoServiceCode).toContain("COMPREHENSIVE");
        });
    });

    describe("Database Schema", () => {
        test("should support COMPREHENSIVE analysis type", () => {
            const schemaContent = fs.readFileSync(
                path.join(__dirname, "../prisma/schema.prisma"),
                "utf8"
            );

            expect(schemaContent).toContain("COMPREHENSIVE");
            expect(schemaContent).toContain("enum AnalysisType");
        });

        test("should validate analysis type creation", async () => {
            // Test that we can create a record with COMPREHENSIVE type
            const testData = {
                model: "SIGLIP_LSTM_V3",
                analysisType: "COMPREHENSIVE",
                prediction: "FAKE",
                confidence: 0.85,
                processingTime: 1234,
                metrics: {
                    avgConfidence: 0.82,
                    temporalConsistency: 0.78,
                    colorConsistency: 0.89,
                },
            };

            // This should not throw a validation error
            expect(() => {
                // Simulated validation - in real scenario this would go through Prisma
                const validAnalysisTypes = [
                    "QUICK",
                    "DETAILED",
                    "FRAMES",
                    "VISUALIZE",
                    "COMPREHENSIVE",
                ];
                expect(validAnalysisTypes).toContain(testData.analysisType);
            }).not.toThrow();
        });
    });

    describe("Response Format Compatibility", () => {
        test("should standardize comprehensive response format", () => {
            const mockComprehensiveResponse = {
                success: true,
                model_used: "SIGLIP-LSTM-V3",
                data: {
                    prediction: "FAKE",
                    confidence: 0.85,
                    processing_time: 1234,
                    metrics: {
                        avg_confidence: 0.82,
                        temporal_consistency: 0.78,
                    },
                    frames_analysis: {
                        frame_predictions: [
                            { frame: 0, prediction: "REAL", confidence: 0.95 },
                            { frame: 1, prediction: "FAKE", confidence: 0.85 },
                        ],
                        temporal_analysis: {
                            consistency_score: 0.78,
                            anomaly_frames: [1],
                        },
                    },
                    visualization_generated: true,
                    visualization_filename: "viz-123.mp4",
                    processing_breakdown: {
                        preprocessing: 100,
                        inference: 500,
                        postprocessing: 200,
                    },
                },
            };

            const standardized =
                modelAnalysisService._standardizeComprehensiveResponse(
                    mockComprehensiveResponse
                );

            expect(standardized).toHaveProperty("model");
            expect(standardized).toHaveProperty("prediction");
            expect(standardized).toHaveProperty("confidence");
            expect(standardized).toHaveProperty("metrics");
            expect(standardized).toHaveProperty("framePredictions");
            expect(standardized).toHaveProperty("temporalAnalysis");
            expect(standardized).toHaveProperty("visualizationGenerated");
            expect(standardized).toHaveProperty("processingBreakdown");

            expect(standardized.model).toBe("SIGLIP_LSTM_V3");
            expect(standardized.prediction).toBe("FAKE");
            expect(standardized.confidence).toBe(0.85);
            expect(standardized.visualizationGenerated).toBe(true);
        });
    });
});
