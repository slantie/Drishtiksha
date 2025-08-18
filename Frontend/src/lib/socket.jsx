// src/lib/socket.jsx

import React from "react";
import { io } from "socket.io-client";
import { queryClient } from "./queryClient";
import { queryKeys } from "./queryKeys";
import { toastOrchestrator } from "./toastOrchestrator.jsx";

const VITE_BACKEND_URL =
    import.meta.env.VITE_BACKEND_URL || "http://localhost:3000";

class SocketService {
    socket = null;

    connect(token) {
        if (this.socket) this.disconnect();
        if (!token) return;

        this.socket = io(VITE_BACKEND_URL, { auth: { token } });
        this.socket.on("connect", () =>
            console.log("[Socket] ✅ Connection established:", this.socket.id)
        );
        this.socket.on("disconnect", (reason) =>
            console.log("[Socket] 🔌 Connection disconnected:", reason)
        );
        this.socket.on("connect_error", (err) =>
            console.error("[Socket] ❌ Connection error:", err.message)
        );
        this.setupEventListeners();
    }

    disconnect() {
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
        }
    }

    setupEventListeners() {
        if (!this.socket) return;

        this.socket.on("progress_update", (progress) => {
            console.log(
                "[Socket] 📩 Received 'progress_update' event:",
                progress
            );
            const { videoId, event, message, data } = progress;

            switch (event) {
                case "PROCESSING_STARTED":
                    toastOrchestrator.startVideoProcessing(videoId, message);
                    break;
                case "ANALYSIS_STARTED":
                case "FRAME_ANALYSIS_PROGRESS":
                case "VISUALIZATION_UPLOADING":
                case "VISUALIZATION_COMPLETED":
                    if (data?.modelName) {
                        toastOrchestrator.updateModelProgress(
                            videoId,
                            data.modelName,
                            message,
                            data.progress,
                            data.total
                        );
                    }
                    break;
                case "ANALYSIS_COMPLETED":
                    if (data?.modelName) {
                        toastOrchestrator.resolveModelProgress(
                            videoId,
                            data.modelName,
                            data.success
                        );
                    }
                    break;
            }

            // Invalidate query to keep the UI on pages like /results/:videoId in sync
            queryClient.invalidateQueries({
                queryKey: queryKeys.videos.detail(videoId),
            });
        });

        this.socket.on("video_update", (video) => {
            console.log(
                "[Socket] 📩 Received final 'video_update' event:",
                video
            );
            toastOrchestrator.resolveVideoProcessing(
                video.id,
                video.filename,
                true
            );

            // Invalidate both detail and list views
            queryClient.invalidateQueries({
                queryKey: queryKeys.videos.detail(video.id),
            });
            queryClient.invalidateQueries({
                queryKey: queryKeys.videos.lists(),
            });
        });

        this.socket.on("processing_error", (errorData) => {
            console.error(
                "[Socket] 📩 Received 'processing_error' event:",
                errorData
            );
            toastOrchestrator.resolveVideoProcessing(
                errorData.videoId,
                "your video",
                false,
                errorData.error
            );

            queryClient.invalidateQueries({
                queryKey: queryKeys.videos.detail(errorData.videoId),
            });
            queryClient.invalidateQueries({
                queryKey: queryKeys.videos.lists(),
            });
        });
    }
}

export const socketService = new SocketService();
