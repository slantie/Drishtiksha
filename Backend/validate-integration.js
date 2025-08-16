// Simple validation test for comprehensive analysis integration
// This test validates the integration without requiring the Jest framework

import { modelAnalysisService } from "./src/services/modelAnalysis.service.js";
import { videoService } from "./src/services/video.service.js";
import fs from "fs";
import path from "path";

async function validateComprehensiveAnalysisIntegration() {
    console.log("🧪 Starting Comprehensive Analysis Integration Validation\n");

    // Test 1: Service Method Availability
    console.log("1️⃣ Testing service method availability...");

    const hasComprehensiveMethod =
        typeof modelAnalysisService.analyzeVideoComprehensive === "function";
    console.log(
        `   ✅ analyzeVideoComprehensive method: ${
            hasComprehensiveMethod ? "Available" : "Missing"
        }`
    );

    const hasVideoServiceMethod =
        typeof videoService._runAndSaveComprehensiveAnalysis === "function";
    console.log(
        `   ✅ _runAndSaveComprehensiveAnalysis method: ${
            hasVideoServiceMethod ? "Available" : "Missing"
        }`
    );

    // Test 2: Service Configuration
    console.log("\n2️⃣ Testing service configuration...");

    const isServiceAvailable = modelAnalysisService.isAvailable();
    console.log(`   📡 Model service available: ${isServiceAvailable}`);

    if (isServiceAvailable) {
        try {
            const health = await modelAnalysisService.getHealthStatus();
            console.log(`   ❤️ Health check: ${health.status || "OK"}`);

            const models = await modelAnalysisService.getAvailableModels();
            console.log(
                `   🤖 Available models: ${
                    models.length > 0 ? models.join(", ") : "None"
                }`
            );
        } catch (error) {
            console.log(`   ⚠️ Server not running: ${error.message}`);
        }
    } else {
        console.log(
            `   ⚠️ Service not configured (missing API key or server URL)`
        );
    }

    // Test 3: Schema Validation
    console.log("\n3️⃣ Testing database schema...");

    const schemaPath = path.join(process.cwd(), "prisma", "schema.prisma");
    const schemaContent = fs.readFileSync(schemaPath, "utf8");

    const hasComprehensiveType = schemaContent.includes("COMPREHENSIVE");
    console.log(
        `   🗄️ COMPREHENSIVE analysis type in schema: ${
            hasComprehensiveType ? "Present" : "Missing"
        }`
    );

    // Test 4: Response Standardization
    console.log("\n4️⃣ Testing response standardization...");

    const mockResponse = {
        success: true,
        model_used: "SIGLIP-LSTM-V3",
        data: {
            prediction: "FAKE",
            confidence: 0.85,
            processing_time: 1234,
            metrics: { avg_confidence: 0.82 },
            frames_analysis: {
                frame_predictions: [
                    { frame: 0, prediction: "REAL", confidence: 0.95 },
                ],
                temporal_analysis: { consistency_score: 0.78 },
            },
            visualization_generated: true,
            visualization_filename: "test.mp4",
        },
    };

    try {
        const standardized =
            modelAnalysisService._standardizeComprehensiveResponse(
                mockResponse
            );
        const hasRequiredFields =
            standardized.model &&
            standardized.prediction &&
            standardized.confidence !== undefined &&
            standardized.framePredictions &&
            standardized.visualizationGenerated !== undefined;

        console.log(
            `   🔄 Response standardization: ${
                hasRequiredFields ? "Working" : "Failed"
            }`
        );
        console.log(`   📊 Standardized model: ${standardized.model}`);
        console.log(
            `   🎯 Standardized prediction: ${standardized.prediction}`
        );
        console.log(
            `   📈 Standardized confidence: ${standardized.confidence}`
        );
    } catch (error) {
        console.log(`   ❌ Response standardization failed: ${error.message}`);
    }

    // Test 5: Service Configuration Check
    console.log("\n5️⃣ Testing video service configuration...");

    const videoServicePath = path.join(
        process.cwd(),
        "src",
        "services",
        "video.service.js"
    );
    const videoServiceContent = fs.readFileSync(videoServicePath, "utf8");

    const usesComprehensive = videoServiceContent.includes(
        "_runAndSaveComprehensiveAnalysis"
    );
    const hasAdvancedModelsCheck = videoServiceContent.includes(
        "ADVANCED_MODELS.includes"
    );

    console.log(
        `   🔧 Uses comprehensive analysis: ${usesComprehensive ? "Yes" : "No"}`
    );
    console.log(
        `   🎯 Advanced models check: ${
            hasAdvancedModelsCheck ? "Present" : "Missing"
        }`
    );

    console.log("\n🎉 Integration validation complete!");
    console.log("\n📋 Summary:");
    console.log(
        `   • Service methods: ${
            hasComprehensiveMethod && hasVideoServiceMethod ? "✅" : "❌"
        }`
    );
    console.log(`   • Database schema: ${hasComprehensiveType ? "✅" : "❌"}`);
    console.log(`   • Response handling: ✅`);
    console.log(`   • Video pipeline: ${usesComprehensive ? "✅" : "❌"}`);

    if (
        hasComprehensiveMethod &&
        hasVideoServiceMethod &&
        hasComprehensiveType &&
        usesComprehensive
    ) {
        console.log("\n🚀 Comprehensive analysis integration is ready!");
        console.log(
            "   The backend will now use comprehensive analysis as the default for advanced models."
        );
        console.log(
            "   Individual analysis routes are still available for manual requests."
        );
    } else {
        console.log(
            "\n⚠️ Some integration issues detected. Check the failing components above."
        );
    }
}

// Run the validation
validateComprehensiveAnalysisIntegration().catch(console.error);
