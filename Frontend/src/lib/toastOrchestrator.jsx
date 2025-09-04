// src/lib/toastOrchestrator.jsx

import React from "react";
import { showToast } from "../utils/toast.jsx";
import { ToastProgress } from "../components/ui/ToastProgress.jsx";

class ToastOrchestrator {
  // These maps now store toasts for any mediaId, not just videos.
  mediaToastMap = new Map(); // Stores the main toast for a media item
  modelToastMap = new Map(); // Stores individual toasts for each model within a media item

  // NEW: A single, generic event handler
  handleProgressEvent(mediaId, event, message, data) {
    // Log all incoming progress events for debugging
    // console.log(`[ToastOrchestrator] Handling event for media ${mediaId}: ${event}`, data);

    switch (event) {
      case "PROCESSING_STARTED":
        // This event is sent when the worker first picks up any job for this media run.
        // It's a good place to start the main media toast.
        this.startMediaProcessing(mediaId, message);
        break;

      case "ANALYSIS_STARTED":
      case "FRAME_ANALYSIS_PROGRESS":
      case "AUDIO_EXTRACTION_START":
      case "SPECTROGRAM_GENERATION_START":
      case "VISUALIZATION_GENERATION_START": // Added this event based on common ML server outputs
      case "VISUALIZATION_UPLOADING":
        // These events all update a model-specific toast
        if (data?.model_name) {
          // Use model_name from Python backend payload
          this.updateModelProgress(
            mediaId,
            data.model_name,
            message,
            data.progress,
            data.total
          );
        }
        break;

      case "ANALYSIS_COMPLETED":
      case "VISUALIZATION_COMPLETED": // Handle visualization completion as well
        if (data?.model_name) {
          this.resolveModelProgress(
            mediaId,
            data.model_name,
            data.success // Expecting success flag from backend
          );
        }
        break;
      case "PROCESSING_FAILED": // New general failure event
        if (data?.model_name) {
          this.resolveModelProgress(
            mediaId,
            data.model_name,
            false,
            data.error_message
          );
        }
        break;
    }
  }

  startMediaProcessing(mediaId, message) {
    // If a main toast already exists for this mediaId, don't create a new one.
    if (this.mediaToastMap.has(mediaId)) return;

    const toastId = showToast.loading(<ToastProgress message={message} />, {
      duration: Infinity,
    });
    this.mediaToastMap.set(mediaId, toastId);
    console.log(
      `[ToastOrchestrator] ✨ Created main toast (ID: ${toastId}) for media ${mediaId}`
    );
  }

  updateModelProgress(mediaId, modelName, message, progress, total) {
    const modelToastKey = `${mediaId}-${modelName}`;
    let toastId = this.modelToastMap.get(modelToastKey);

    const toastContent = (
      <ToastProgress
        modelName={modelName}
        message={message}
        progress={progress}
        total={total}
      />
    );

    if (!toastId) {
      toastId = showToast.loading(toastContent, { duration: Infinity });
      this.modelToastMap.set(modelToastKey, toastId);
      console.log(
        `[ToastOrchestrator] 🚀 Started model toast (ID: ${toastId}) for ${modelName} on media ${mediaId}`
      );
    } else {
      showToast.loading(toastContent, { id: toastId }); // Update existing toast
    }
  }

  resolveModelProgress(mediaId, modelName, success, errorMsg = "") {
    const modelToastKey = `${mediaId}-${modelName}`;
    const toastId = this.modelToastMap.get(modelToastKey);

    if (toastId) {
      const message = success
        ? `Analysis for '${modelName}' complete.`
        : `Analysis for '${modelName}' failed: ${errorMsg || "Unknown error"}.`;
      const toastFn = success ? showToast.success : showToast.error;

      toastFn(<ToastProgress modelName={modelName} message={message} />, {
        id: toastId,
        duration: 5000,
      });
      this.modelToastMap.delete(modelToastKey);
      console.log(
        `[ToastOrchestrator] ✅/❌ Resolved model toast (ID: ${toastId}) for ${modelName} on media ${mediaId} with status: ${success}`
      );
    }
  }

  // This method is called when the *final* 'media_update' or 'processing_error' event is received
  // from the backend, signifying the entire workflow for a media item is complete (or failed).
  resolveMediaProcessing(mediaId, filename, success, errorMsg = "") {
    const mainToastId = this.mediaToastMap.get(mediaId);
    if (mainToastId) {
      const message = success
        ? `All analyses for "${filename}" are complete!`
        : `Processing for "${filename}" failed: ${
            errorMsg || "Please check media status for details."
          }`;
      const toastFn = success ? showToast.success : showToast.error;

      toastFn(<ToastProgress message={message} />, {
        id: mainToastId,
        duration: 8000,
      });
      this.mediaToastMap.delete(mediaId);
      console.log(
        `[ToastOrchestrator] 🎉/😭 Resolved main toast (ID: ${mainToastId}) for media ${mediaId} with status: ${success}`
      );
    }

    // Also ensure all individual model toasts for this media are dismissed/resolved
    this.modelToastMap.forEach((toastId, key) => {
      if (key.startsWith(`${mediaId}-`)) {
        showToast.dismiss(toastId);
        this.modelToastMap.delete(key);
        console.log(
          `[ToastOrchestrator] Dismissed lingering model toast (ID: ${toastId}) for key: ${key}`
        );
      }
    });
  }
}

export const toastOrchestrator = new ToastOrchestrator();
