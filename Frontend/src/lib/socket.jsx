// src/lib/socket.jsx

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
            console.log("[Socket] 📩 Received 'progress_update' event:", progress);
            
            // The backend uses 'videoId' as the key, which is our generic mediaId.
            const mediaId = progress.videoId || progress.mediaId;
            const { event, message, data } = progress;
            
            if (!mediaId) return;

            // Delegate all toast logic to the orchestrator
            toastOrchestrator.handleProgressEvent(mediaId, event, message, data);

            // Invalidate the specific media item's query to trigger a refetch in the UI
            queryClient.invalidateQueries({
                queryKey: queryKeys.media.detail(mediaId),
            });
        });

        // REFACTORED: The 'video_update' event now represents a generic media update.
        this.socket.on("video_update", (media) => {
            console.log("[Socket] 📩 Received final 'media_update' event:", media);
            
            // The backend still calls the event 'video_update', but the payload is a generic media object.
            const mediaId = media.id;
            const filename = media.filename || "your media file";

            toastOrchestrator.resolveVideoProcessing(mediaId, filename, true);

            // Invalidate both the specific media item and the list of all media
            queryClient.invalidateQueries({
                queryKey: queryKeys.media.detail(mediaId),
            });
            queryClient.invalidateQueries({
                queryKey: queryKeys.media.lists(),
            });
        });

        this.socket.on("processing_error", (errorData) => {
            console.error("[Socket] 📩 Received 'processing_error' event:", errorData);

            const mediaId = errorData.videoId || errorData.mediaId;
            if (!mediaId) return;

            toastOrchestrator.resolveVideoProcessing(mediaId, "your media file", false, errorData.error);
            
            queryClient.invalidateQueries({ queryKey: queryKeys.media.detail(mediaId) });
            queryClient.invalidateQueries({ queryKey: queryKeys.media.lists() });
        });
    }
}

export const socketService = new SocketService();