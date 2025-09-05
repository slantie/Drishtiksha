// scripts/clear-queue.js

import { redisConnection } from "../src/config/queue.js";
import { Queue } from "bullmq";

async function clearQueue() {
    console.log("🧹 Clearing the media processing queue...");

    const queue = new Queue(process.env.MEDIA_PROCESSING_QUEUE_NAME || "media-processing-queue", {
        connection: redisConnection,
    });

    try {
        // Clear all jobs from the queue
        await queue.drain();

        // Clean up completed and failed jobs
        await queue.clean(0, 1000, "completed");
        await queue.clean(0, 1000, "failed");

        console.log("✅ Queue cleared successfully");

        // Get current job count
        const waiting = await queue.getWaiting();
        const active = await queue.getActive();
        const completed = await queue.getCompleted();
        const failed = await queue.getFailed();

        console.log(`📊 Queue status:`);
        console.log(`   Waiting: ${waiting.length}`);
        console.log(`   Active: ${active.length}`);
        console.log(`   Completed: ${completed.length}`);
        console.log(`   Failed: ${failed.length}`);
    } catch (error) {
        console.error("❌ Error clearing queue:", error.message);
    } finally {
        await queue.close();
        process.exit(0);
    }
}

clearQueue();
