// Backend/server.js

import dotenv from "dotenv";
import { createServer } from "http";
import { app } from "./src/app.js";
import { connectDatabase, disconnectDatabase } from "./src/config/database.js";
import { initializeSocketIO } from "./src/config/socket.js";
import { QueueEvents } from "bullmq";
import { VIDEO_PROCESSING_QUEUE_NAME } from "./src/config/constants.js";
import { videoRepository } from "./src/repositories/video.repository.js";
import { eventService } from "./src/services/event.service.js";
import logger from "./src/utils/logger.js";

dotenv.config({ path: "./.env" });

const PORT = process.env.PORT || 4000;
const httpServer = createServer(app);

const io = initializeSocketIO(httpServer);
app.set("io", io);

// --- BullMQ Event Listener ---
const queueEvents = new QueueEvents(VIDEO_PROCESSING_QUEUE_NAME, {
    connection: {
        host: process.env.REDIS_URL
            ? new URL(process.env.REDIS_URL).hostname
            : "localhost",
        port: process.env.REDIS_URL
            ? parseInt(new URL(process.env.REDIS_URL).port)
            : 6379,
    },
});

// CORRECTED: Reverted to the 'completed' event, which is more reliable.
// ADDED: Robust logic to specifically identify the finalizer job.
// REASON: This is the definitive fix. It ensures we only act when the entire workflow is
// verifiably complete, and then sends the final, correct state to the client.
queueEvents.on("completed", async ({ jobId }) => {
    if (jobId.endsWith("-finalizer")) {
        const videoId = jobId.replace("-finalizer", "");
        logger.info(
            `[QueueEvents] Finalizer Job for video ${videoId} has completed.`
        );
        const video = await videoRepository.findById(videoId);
        if (video) {
            io.to(video.userId).emit("video_update", video);
            logger.info(
                `[SocketIO] Emitted final 'video_update' for video ${videoId} to user ${video.userId}.`
            );
        }
    }
});

queueEvents.on("failed", async ({ jobId, failedReason }) => {
    logger.error(`[QueueEvents] Job ${jobId} failed: ${failedReason}`);
    const videoId = jobId.split("-")[0];

    try {
        const video = await videoRepository.findById(videoId);
        if (video) {
            io.to(video.userId).emit("processing_error", {
                videoId,
                error: failedReason,
            });
            io.to(video.userId).emit("video_update", video);
            logger.info(
                `[SocketIO] Emitted 'processing_error' for failed job ${jobId} to user ${video.userId}.`
            );
        }
    } catch (error) {
        logger.error(
            `[QueueEvents] Error fetching video ${videoId} after failure: ${error.message}`
        );
    }
});
// --- End BullMQ Event Listener ---

eventService.listenForProgress((progressData) => {
    if (progressData.userId) {
        io.to(progressData.userId).emit("progress_update", progressData);
        logger.info(
            `[SocketIO] Emitted 'progress_update' (${progressData.event}) for video ${progressData.videoId} to user ${progressData.userId}.`
        );
    }
});

const startServer = async () => {
    try {
        await connectDatabase();
        httpServer.listen(PORT, () => {
            console.log(`\n🚀 Server is running at: http://localhost:${PORT}`);
            console.log(
                `   Environment: ${process.env.NODE_ENV || "development"}`
            );
        });
        const shutdown = async (signal) => {
            console.log(`\n${signal} received. Shutting down gracefully...`);
            await queueEvents.close();
            httpServer.close(async () => {
                await disconnectDatabase();
                console.log("🔌 Server and database connections closed.");
                process.exit(0);
            });
        };
        process.on("SIGTERM", () => shutdown("SIGTERM"));
        process.on("SIGINT", () => shutdown("SIGINT"));
    } catch (error) {
        console.error("❌ Failed to start server:", error);
        process.exit(1);
    }
};

startServer();
