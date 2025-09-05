// src/lib/socket.jsx

import { io } from "socket.io-client";
import { queryClient } from "./queryClient";
import { queryKeys } from "./queryKeys";
import { toastOrchestrator } from "./toastOrchestrator.jsx";
import { config } from "../config/env.js";

class SocketService {
  socket = null;

  connect(token) {
    if (this.socket?.connected) {
      console.log(
        "[Socket] Already connected, skipping new connection attempt."
      );
      return;
    }
    if (!token) {
      console.log("[Socket] No token provided, skipping socket connection.");
      return;
    }

    console.log("[Socket] Attempting to connect...");
    this.socket = io(config.VITE_BACKEND_URL, { auth: { token } }); // Use validated config

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
      console.log("[Socket] Forcibly disconnecting...");
      this.socket.disconnect();
      this.socket = null;
    }
  }

  setupEventListeners() {
    if (!this.socket) {
      console.warn("[Socket] No socket instance to setup event listeners.");
      return;
    }

    this.socket.off("progress_update"); // Remove old listeners to prevent duplicates
    this.socket.off("media_update"); // Corrected event name
    this.socket.off("processing_error");

    this.socket.on("progress_update", (progress) => {
      console.log("[Socket] 📩 Received 'progress_update' event:", progress);

      // The backend events now consistently use 'mediaId'
      const mediaId = progress.mediaId;
      const { event, message, data } = progress;

      if (!mediaId) {
        console.warn(
          "[Socket] 'progress_update' event received without mediaId:",
          progress
        );
        return;
      }

      // Delegate all toast logic to the orchestrator
      toastOrchestrator.handleProgressEvent(mediaId, event, message, data);

      // Invalidate the specific media item's query to trigger a refetch in the UI
      queryClient.invalidateQueries({
        queryKey: queryKeys.media.detail(mediaId),
      });
    });

    // REFACTORED: Listen for 'media_update' from the backend.
    this.socket.on("media_update", (media) => {
      console.log("[Socket] 📩 Received final 'media_update' event:", media);

      // The backend now sends a generic media object with `id`
      const mediaId = media.id;
      const filename = media.filename || "your media file";

      toastOrchestrator.resolveMediaProcessing(mediaId, filename, true);

      // Invalidate both the specific media item and the list of all media
      queryClient.invalidateQueries({
        queryKey: queryKeys.media.detail(mediaId),
      });
      queryClient.invalidateQueries({
        queryKey: queryKeys.media.lists(),
      });
    });

    this.socket.on("processing_error", (errorData) => {
      console.error(
        "[Socket] 📩 Received 'processing_error' event:",
        errorData
      );

      const mediaId = errorData.mediaId; // Use mediaId consistent with backend
      const filename = errorData.filename || "your media file"; // Pass filename if available

      if (!mediaId) {
        console.warn(
          "[Socket] 'processing_error' event received without mediaId:",
          errorData
        );
        return;
      }

      toastOrchestrator.resolveMediaProcessing(
        mediaId,
        filename,
        false,
        errorData.error
      );

      queryClient.invalidateQueries({
        queryKey: queryKeys.media.detail(mediaId),
      });
      queryClient.invalidateQueries({ queryKey: queryKeys.media.lists() });
    });
  }
}

export const socketService = new SocketService();
