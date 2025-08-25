import { Worker } from "bullmq";
import { redis } from "../src/config/redis.js";
import { processVideo } from "../src/workers/media.worker.js";

async function restartWorker() {
    console.log("🔄 Starting media processing worker...");

    try {
        // Create and start the worker
        const worker = new Worker("media-processing", processVideo, {
            connection: redis,
            concurrency: 1,
            removeOnComplete: 10,
            removeOnFail: 50,
        });

        worker.on("completed", (job) => {
            console.log(`✅ Job ${job.id} completed successfully`);
        });

        worker.on("failed", (job, err) => {
            console.log(`❌ Job ${job.id} failed: ${err.message}`);
        });

        worker.on("error", (err) => {
            console.error("Worker error:", err);
        });

        console.log("✅ Worker started successfully");
        console.log("Press Ctrl+C to stop the worker");

        // Keep the process running
        process.on("SIGINT", () => {
            console.log("\n🛑 Stopping worker...");
            worker.close();
            process.exit(0);
        });
    } catch (error) {
        console.error("❌ Failed to start worker:", error.message);
        process.exit(1);
    }
}

restartWorker();
