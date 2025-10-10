// src/lib/toastOrchestrator.jsx

import React from "react";
import { showToast } from "../utils/toast.jsx";
import { ToastProgress } from "../components/ui/ToastProgress.jsx";

class ToastOrchestrator {
  // Updated to match legacy code structure
  mediaToastMap = new Map(); // videoId -> toastId (main processing toast)
  modelToastMap = new Map(); // "mediaId-modelName" -> toastId (individual model toasts)
  progressCallbacks = new Map(); // For detailed progress tracking

  // Register a callback for detailed progress updates
  registerProgressCallback(mediaId, callback) {
    this.progressCallbacks.set(mediaId, callback);
  }

  unregisterProgressCallback(mediaId) {
    this.progressCallbacks.delete(mediaId);
  }

  // Main entry point - matches legacy handleProgressEvent
  handleProgressEvent(mediaId, event, message, data) {
    console.log(`[ToastOrchestrator] Handling event: ${event}`, {
      mediaId,
      message,
      data,
    });

    // Call registered callback for detailed progress tracking
    const callback = this.progressCallbacks.get(mediaId);
    if (callback) {
      callback({ event, message, data, mediaId });
    }

    // Match the exact event names from your legacy code
    switch (event) {
      case "PROCESSING_STARTED":
        this.startMediaProcessing(mediaId, message);
        break;

      // This is the key event that was missing!
      case "FRAME_ANALYSIS_PROGRESS":
      case "ANALYSIS_STARTED":
      case "VISUALIZATION_GENERATION_PROGRESS":
      case "WINDOW_PROCESSING_PROGRESS":
      case "AUDIO_EXTRACTION_START":
      case "AUDIO_EXTRACTION_COMPLETE":
      case "SPECTROGRAM_GENERATION_START":
      case "SPECTROGRAM_GENERATION_COMPLETE":
        if (data?.model_name || data?.modelName) {
          const modelName = data.model_name || data.modelName;
          this.updateModelProgress(
            mediaId,
            modelName,
            message,
            data.progress,
            data.total,
            data // Pass the full data object to include details
          );
        }
        break;

      case "ANALYSIS_COMPLETED":
      case "ANALYSIS_COMPLETE":
        if (data?.model_name || data?.modelName) {
          const modelName = data.model_name || data.modelName;
          this.resolveModelProgress(mediaId, modelName, true);
        }
        break;

      case "ANALYSIS_FAILED":
      case "PROCESSING_FAILED":
        if (data?.model_name || data?.modelName) {
          const modelName = data.model_name || data.modelName;
          this.resolveModelProgress(
            mediaId,
            modelName,
            false,
            data.error_message || data.error || "Analysis failed"
          );
        }
        break;

      default:
        console.log(`[ToastOrchestrator] Unhandled event type: ${event}`, {
          mediaId,
          message,
          data,
        });
        break;
    }
  }

  // Legacy method names for compatibility
  startVideoProcessing(mediaId, _message) {
    return this.startMediaProcessing(mediaId, _message);
  }

  startMediaProcessing(mediaId, _message) {
    if (this.mediaToastMap.has(mediaId)) {
      console.log(
        `[ToastOrchestrator] Media toast already exists for ${mediaId}`
      );
      return;
    }

    // SUPPRESSED: No toast spam - ProgressPanel handles this
    // Instead, just mark that we're tracking this media
    this.mediaToastMap.set(mediaId, 'progress-panel-tracking');
    console.log(
      `[ToastOrchestrator] ✅ Started tracking media ${mediaId} (no toast - using ProgressPanel)`
    );
  }

  updateModelProgress(
    mediaId,
    modelName,
    message,
    progress,
    total,
    _details = {}
  ) {
    const modelToastKey = `${mediaId}-${modelName}`;

    // SUPPRESSED: No individual model toasts - ProgressPanel handles this
    // Just log the progress for debugging
    if (progress && total) {
      const percentage = ((progress / total) * 100).toFixed(1);
      console.log(
        `[ToastOrchestrator] 🔄 Model progress for ${modelName}: ${percentage}% (${progress}/${total})`
      );
    } else {
      console.log(
        `[ToastOrchestrator] 🔄 Model update for ${modelName}: ${message}`
      );
    }

    // Mark that we've seen this model (for cleanup purposes)
    if (!this.modelToastMap.has(modelToastKey)) {
      this.modelToastMap.set(modelToastKey, 'progress-panel-tracking');
    }
  }

  resolveModelProgress(mediaId, modelName, success, errorMsg = "") {
    const modelToastKey = `${mediaId}-${modelName}`;

    // SUPPRESSED: No individual model completion toasts - ProgressPanel handles this
    console.log(
      `[ToastOrchestrator] ✅ Model ${modelName} ${success ? 'completed' : 'failed'}${
        errorMsg ? `: ${errorMsg}` : ""
      }`
    );

    // Clean up tracking
    this.modelToastMap.delete(modelToastKey);
  }

  resolveMediaProcessing(mediaId, filename, success, errorMsg = "") {
    // Show ONLY ONE toast for final completion/failure
    let message;
    if (success) {
      message = `🎉 Analysis complete for "${filename}"!`;
    } else {
      message = `❌ Analysis failed for "${filename}"${
        errorMsg ? `: ${errorMsg}` : ""
      }`;
    }

    const toastContent = <ToastProgress message={message} />;
    const toastFn = success ? showToast.success : showToast.error;
    toastFn(toastContent, {
      duration: success ? 5000 : 8000,
    });

    console.log(
      `[ToastOrchestrator] ✅ Media processing ${success ? 'completed' : 'failed'}: ${filename}`
    );

    // Clean up tracking
    this.mediaToastMap.delete(mediaId);

    // Clean up any remaining model tracking for this media
    this.modelToastMap.forEach((_toastId, key) => {
      if (key.startsWith(`${mediaId}-`)) {
        this.modelToastMap.delete(key);
      }
    });

    // Clean up progress callback
    this.unregisterProgressCallback(mediaId);
  }

  // Legacy compatibility methods
  resolveVideoProcessing(mediaId, filename, success, errorMsg = "") {
    return this.resolveMediaProcessing(mediaId, filename, success, errorMsg);
  }
}

export const toastOrchestrator = new ToastOrchestrator();

// Hook to initialize the toast orchestrator (for App.jsx or main components)
export const useToastOrchestrator = () => {
  // This hook can be used to initialize any global toast orchestrator logic
  // Currently just returns the orchestrator instance
  return toastOrchestrator;
};
