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
      case "ANALYSIS_STARTED": // Also handle analysis started
        if (data?.model_name || data?.modelName) {
          const modelName = data.model_name || data.modelName;
          this.updateModelProgress(
            mediaId,
            modelName,
            message,
            data.progress,
            data.total,
            data
          );
        }
        break;

      case "WINDOW_PROCESSING_PROGRESS":
        if (data?.model_name || data?.modelName) {
          const modelName = data.model_name || data.modelName;
          this.updateModelProgress(
            mediaId,
            modelName,
            message,
            data.progress,
            data.total,
            data
          );
        }
        break;

      case "VISUALIZATION_GENERATION_PROGRESS":
        if (data?.model_name || data?.modelName) {
          const modelName = data.model_name || data.modelName;
          this.updateModelProgress(
            mediaId,
            modelName,
            message,
            data.progress,
            data.total,
            data
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

      // Audio processing events
      case "AUDIO_EXTRACTION_START":
        if (data?.model_name || data?.modelName) {
          const modelName = data.model_name || data.modelName;
          this.updateModelProgress(
            mediaId,
            modelName,
            "Extracting audio from media file...",
            null,
            null,
            { phase: "audio_extraction" }
          );
        }
        break;

      case "AUDIO_EXTRACTION_COMPLETE":
        if (data?.model_name || data?.modelName) {
          const modelName = data.model_name || data.modelName;
          this.updateModelProgress(
            mediaId,
            modelName,
            "Audio extraction completed",
            null,
            null,
            { phase: "audio_extracted" }
          );
        }
        break;

      case "SPECTROGRAM_GENERATION_START":
        if (data?.model_name || data?.modelName) {
          const modelName = data.model_name || data.modelName;
          this.updateModelProgress(
            mediaId,
            modelName,
            "Generating spectrogram visualization...",
            null,
            null,
            { phase: "spectrogram_generation" }
          );
        }
        break;

      case "SPECTROGRAM_GENERATION_COMPLETE":
        if (data?.model_name || data?.modelName) {
          const modelName = data.model_name || data.modelName;
          this.updateModelProgress(
            mediaId,
            modelName,
            "Spectrogram generation completed",
            null,
            null,
            { phase: "spectrogram_complete" }
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
  startVideoProcessing(mediaId, message) {
    return this.startMediaProcessing(mediaId, message);
  }

  startMediaProcessing(mediaId, message) {
    if (this.mediaToastMap.has(mediaId)) {
      console.log(
        `[ToastOrchestrator] Media toast already exists for ${mediaId}`
      );
      return;
    }

    const toastContent = <ToastProgress message={message} />;

    const toastId = showToast.loading(toastContent, { duration: Infinity });
    this.mediaToastMap.set(mediaId, toastId);
    console.log(
      `[ToastOrchestrator] ✨ Created main toast (ID: ${toastId}) for media ${mediaId}`
    );
  }

  updateModelProgress(
    mediaId,
    modelName,
    message,
    progress,
    total,
    details = {}
  ) {
    const modelToastKey = `${mediaId}-${modelName}`;
    let toastId = this.modelToastMap.get(modelToastKey);

    // Create enhanced message with progress info
    let displayMessage = message;

    // Add detailed progress information
    if (progress && total) {
      const percentage = ((progress / total) * 100).toFixed(1);
      displayMessage = `${message} (${progress}/${total} - ${percentage}%)`;

      // Add specific details if available
      if (details.faces_detected_so_far) {
        displayMessage += ` - ${details.faces_detected_so_far} faces detected`;
      }
      if (details.windows_processed) {
        displayMessage += ` - ${details.windows_processed} windows processed`;
      }
    }

    const toastContent = (
      <ToastProgress
        modelName={modelName}
        message={displayMessage}
        progress={progress}
        total={total}
        details={details}
      />
    );

    if (!toastId) {
      toastId = showToast.loading(toastContent, { duration: Infinity });
      this.modelToastMap.set(modelToastKey, toastId);
      console.log(
        `[ToastOrchestrator] ✨ Created model toast (ID: ${toastId}) for ${modelName}`
      );
    } else {
      showToast.loading(toastContent, { id: toastId, duration: Infinity });
      console.log(
        `[ToastOrchestrator] 🔄 Updated model toast (ID: ${toastId}) for ${modelName}: ${displayMessage}`
      );
    }
  }

  resolveModelProgress(mediaId, modelName, success, errorMsg = "") {
    const modelToastKey = `${mediaId}-${modelName}`;
    const toastId = this.modelToastMap.get(modelToastKey);

    if (toastId) {
      let message;
      if (success) {
        message = `✅ ${modelName} analysis completed successfully`;
      } else {
        message = `❌ ${modelName} analysis failed${
          errorMsg ? `: ${errorMsg}` : ""
        }`;
      }

      const toastContent = (
        <ToastProgress modelName={modelName} message={message} />
      );

      const toastFn = success ? showToast.success : showToast.error;
      toastFn(toastContent, { id: toastId, duration: success ? 5000 : 8000 });

      this.modelToastMap.delete(modelToastKey);
      console.log(
        `[ToastOrchestrator] ✅ Resolved model toast (ID: ${toastId}) for ${modelName}`
      );
    }
  }

  resolveMediaProcessing(mediaId, filename, success, errorMsg = "") {
    const mainToastId = this.mediaToastMap.get(mediaId);

    if (mainToastId) {
      let message;
      if (success) {
        message = `🎉 All analyses completed for "${filename}"!`;
      } else {
        message = `❌ Processing failed for "${filename}"${
          errorMsg ? `: ${errorMsg}` : ""
        }`;
      }

      const toastContent = <ToastProgress message={message} />;

      const toastFn = success ? showToast.success : showToast.error;
      toastFn(toastContent, {
        id: mainToastId,
        duration: success ? 8000 : 10000,
      });

      this.mediaToastMap.delete(mediaId);
      console.log(
        `[ToastOrchestrator] ✅ Resolved main media toast (ID: ${mainToastId})`
      );
    }

    // Clean up any remaining model-specific toasts for this media
    this.modelToastMap.forEach((toastId, key) => {
      if (key.startsWith(`${mediaId}-`)) {
        showToast.dismiss(toastId);
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
