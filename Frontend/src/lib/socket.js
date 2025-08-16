// src/lib/socket.js

import { io } from "socket.io-client";
import { queryClient } from "./queryClient";
import { queryKeys } from "./queryKeys";
import { showToast } from "../utils/toast";

const VITE_BACKEND_URL =
    import.meta.env.VITE_BACKEND_URL || "http://localhost:3000";

class SocketService {
    socket = null;

    connect() {
        if (this.socket && this.socket.connected) {
            return;
        }

        const token =
            sessionStorage.getItem("authToken") ||
            localStorage.getItem("authToken");

        if (!token) {
            console.error(
                "Socket.io: No auth token found. Connection aborted."
            );
            return;
        }

        console.log(`Socket.io: Attempting to connect to ${VITE_BACKEND_URL}`);
        this.socket = io(VITE_BACKEND_URL, {
            auth: { token },
        });

        this.socket.on("connect", () => {
            console.log("✅ Socket.io connected:", this.socket.id);
        });

        this.socket.on("disconnect", (reason) => {
            console.log("🔌 Socket.io disconnected:", reason);
        });

        this.socket.on("connect_error", (err) => {
            console.error("Socket.io connection error:", err.message);
        });

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

        this.socket.on("progress_update", (progressData) => {
            console.log("[Socket] Received progress_update:", progressData);
            const videoId = progressData.videoId;

            queryClient.setQueryData(
                queryKeys.videos.detail(videoId),
                (oldData) => {
                    if (!oldData) return;
                    return { ...oldData, status: "PROCESSING" };
                }
            );

            queryClient.setQueryData(queryKeys.videos.lists(), (oldData) => {
                if (!oldData?.data) return oldData;
                return {
                    ...oldData,
                    data: oldData.data.map((v) =>
                        v.id === videoId ? { ...v, status: "PROCESSING" } : v
                    ),
                };
            });
        });

        this.socket.on("video_update", (updatedVideo) => {
            console.log("[Socket] Received video_update:", updatedVideo);
            showToast.success(
                `Analysis for "${updatedVideo.filename}" is complete!`
            );
            queryClient.invalidateQueries({
                queryKey: queryKeys.videos.detail(updatedVideo.id),
            });
            queryClient.invalidateQueries({
                queryKey: queryKeys.videos.lists(),
            });
        });

        this.socket.on("processing_error", (errorData) => {
            console.error("[Socket] Received processing_error:", errorData);
            showToast.error(`Analysis failed: ${errorData.error}`);
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
