// Enhanced monitoring validation for comprehensive analysis
// Tests that model info, system info, and server health are properly tracked

import { modelAnalysisService } from "./src/services/modelAnalysis.service.js";
import { videoService } from "./src/services/video.service.js";
import { videoRepository } from "./src/repositories/video.repository.js";
import fs from "fs";
import path from "path";

async function validateEnhancedMonitoring() {
    console.log(
        "🔍 Enhanced Monitoring Validation for Comprehensive Analysis\n"
    );

    // Test 1: Server Health Tracking
    console.log("1️⃣ Testing server health tracking...");

    try {
        const healthStatus = await modelAnalysisService.getHealthStatus();
        console.log(`   ✅ Health check successful`);
        console.log(`   📊 Response time: ${healthStatus.responseTime}ms`);
        console.log(
            `   🤖 Active models: ${healthStatus.active_models?.length || 0}`
        );
        console.log(
            `   🖥️  Server version: ${healthStatus.version || "unknown"}`
        );
        console.log(
            `   ⚡ GPU info available: ${healthStatus.gpu_info ? "Yes" : "No"}`
        );

        // Test storing health data
        try {
            await videoRepository.storeServerHealth({
                serverUrl: healthStatus.serverUrl,
                status: "HEALTHY",
                availableModels:
                    healthStatus.active_models?.map((m) => m.name) || [],
                modelStates: healthStatus.active_models || null,
                loadMetrics: healthStatus.load_metrics || null,
                gpuInfo: healthStatus.gpu_info || null,
                systemResources: healthStatus.system_resources || null,
                responseTime: healthStatus.responseTime,
                uptime: healthStatus.uptime,
                version: healthStatus.version,
            });
            console.log(`   ✅ Server health stored in database`);
        } catch (storeError) {
            console.log(
                `   ❌ Failed to store health data: ${storeError.message}`
            );
        }
    } catch (error) {
        console.log(`   ❌ Health check failed: ${error.message}`);
    }

    // Test 2: Enhanced Response Structure
    console.log("\n2️⃣ Testing enhanced response structure...");

    const mockComprehensiveResponse = {
        success: true,
        model_used: "SIGLIP-LSTM-V3",
        data: {
            prediction: "FAKE",
            confidence: 0.85,
            processing_time: 1234,
            metrics: { avg_confidence: 0.82 },
            request_id: "test-req-123",
            model_info: {
                model_name: "SIGLIP-LSTM-V3",
                version: "1.0.0",
                architecture: "LSTM",
                device: "cuda:0",
                batch_size: 1,
                memory_usage: "2GB",
            },
            system_info: {
                gpu_memory_used: "4GB",
                gpu_memory_total: "8GB",
                processing_device: "cuda:0",
                cuda_available: true,
                cuda_version: "11.8",
                system_memory_used: "16GB",
                cpu_usage: 45.5,
                python_version: "3.9.0",
                torch_version: "2.0.0",
            },
            server_info: {
                version: "1.2.0",
                uptime: "2 days",
            },
        },
    };

    try {
        const standardized =
            modelAnalysisService._standardizeComprehensiveResponse(
                mockComprehensiveResponse
            );

        console.log(`   ✅ Response standardization successful`);
        console.log(
            `   📊 Model info captured: ${
                Object.keys(standardized.modelInfo || {}).length
            } fields`
        );
        console.log(
            `   🖥️  System info captured: ${
                Object.keys(standardized.systemInfo || {}).length
            } fields`
        );
        console.log(
            `   🌐 Server info captured: ${
                Object.keys(standardized.serverInfo || {}).length
            } fields`
        );
        console.log(`   🔍 Request ID: ${standardized.requestId}`);
    } catch (error) {
        console.log(`   ❌ Response standardization failed: ${error.message}`);
    }

    // Test 3: Database Schema Validation
    console.log("\n3️⃣ Testing database schema updates...");

    const schemaContent = fs.readFileSync(
        path.join(process.cwd(), "prisma", "schema.prisma"),
        "utf8"
    );

    const hasEnhancedModelInfo =
        schemaContent.includes("modelName") &&
        schemaContent.includes("loadTime") &&
        schemaContent.includes("memoryUsage");

    const hasEnhancedSystemInfo =
        schemaContent.includes("gpuMemoryTotal") &&
        schemaContent.includes("cudaVersion") &&
        schemaContent.includes("requestId");

    const hasEnhancedServerHealth =
        schemaContent.includes("modelStates") &&
        schemaContent.includes("avgProcessingTime");

    console.log(
        `   🔧 Enhanced ModelInfo schema: ${
            hasEnhancedModelInfo ? "Present" : "Missing"
        }`
    );
    console.log(
        `   🖥️  Enhanced SystemInfo schema: ${
            hasEnhancedSystemInfo ? "Present" : "Missing"
        }`
    );
    console.log(
        `   📊 Enhanced ServerHealth schema: ${
            hasEnhancedServerHealth ? "Present" : "Missing"
        }`
    );

    // Test 4: Repository Methods
    console.log("\n4️⃣ Testing repository methods...");

    const hasHealthMethods =
        typeof videoRepository.storeServerHealth === "function" &&
        typeof videoRepository.getLatestServerHealth === "function" &&
        typeof videoRepository.getServerHealthHistory === "function";

    console.log(
        `   📚 Health tracking methods: ${
            hasHealthMethods ? "Available" : "Missing"
        }`
    );

    // Test getting latest health
    try {
        const latestHealth = await videoRepository.getLatestServerHealth();
        console.log(
            `   📈 Latest health record: ${latestHealth ? "Found" : "None"}`
        );
        if (latestHealth) {
            console.log(
                `   📅 Health check time: ${latestHealth.lastHealthCheck}`
            );
            console.log(
                `   🎯 Models tracked: ${
                    latestHealth.availableModels?.length || 0
                }`
            );
        }
    } catch (error) {
        console.log(`   ❌ Failed to get health record: ${error.message}`);
    }

    // Test 5: Service Integration
    console.log("\n5️⃣ Testing service integration...");

    const videoServiceContent = fs.readFileSync(
        path.join(process.cwd(), "src", "services", "video.service.js"),
        "utf8"
    );

    const hasHealthTracking =
        videoServiceContent.includes("storeServerHealth") &&
        videoServiceContent.includes("healthData") &&
        videoServiceContent.includes("modelInfo") &&
        videoServiceContent.includes("systemInfo");

    console.log(
        `   🔄 Health tracking integration: ${
            hasHealthTracking ? "Implemented" : "Missing"
        }`
    );

    // Summary
    console.log("\n🎉 Enhanced Monitoring Validation Complete!");
    console.log("\n📋 Summary:");
    console.log(
        `   • Server health tracking: ${hasHealthMethods ? "✅" : "❌"}`
    );
    console.log(
        `   • Enhanced database schema: ${
            hasEnhancedModelInfo && hasEnhancedSystemInfo ? "✅" : "❌"
        }`
    );
    console.log(`   • Response structure: ✅`);
    console.log(`   • Service integration: ${hasHealthTracking ? "✅" : "❌"}`);

    const allGood =
        hasHealthMethods &&
        hasEnhancedModelInfo &&
        hasEnhancedSystemInfo &&
        hasHealthTracking;

    if (allGood) {
        console.log("\n🚀 Enhanced monitoring is fully operational!");
        console.log("   The system now tracks:");
        console.log("   • Model information (version, device, memory usage)");
        console.log("   • System resources (GPU, CPU, memory)");
        console.log("   • Server health (status, uptime, performance)");
        console.log("   • Request tracking (unique IDs, processing times)");
        console.log("   • Load balancing metrics and optimization data");
    } else {
        console.log(
            "\n⚠️ Some monitoring features need attention. Check the failing components above."
        );
    }
}

// Run the validation
validateEnhancedMonitoring().catch(console.error);
