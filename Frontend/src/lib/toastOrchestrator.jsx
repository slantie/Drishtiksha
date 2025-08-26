// src/lib/toastOrchestrator.jsx

import React from "react";
import { showToast } from "../utils/toast.js";
import { ToastProgress } from "../components/ui/ToastProgress.jsx";

class ToastOrchestrator {
    // These maps now store toasts for any mediaId, not just videos.
    mediaToastMap = new Map();
    modelToastMap = new Map();

    // NEW: A single, generic event handler
    handleProgressEvent(mediaId, event, message, data) {
        switch (event) {
            case "PROCESSING_STARTED":
                this.startMediaProcessing(mediaId, message);
                break;
            
            // These events all update a model-specific toast
            case "ANALYSIS_STARTED":
            case "FRAME_ANALYSIS_PROGRESS":
            case "AUDIO_EXTRACTION_START": // Handle new audio events
            case "SPECTROGRAM_GENERATION_START":
            case "VISUALIZATION_UPLOADING":
            case "VISUALIZATION_COMPLETED":
                if (data?.modelName) {
                    this.updateModelProgress(
                        mediaId,
                        data.modelName,
                        message,
                        data.progress,
                        data.total
                    );
                }
                break;
            
            case "ANALYSIS_COMPLETED":
                if (data?.modelName) {
                    this.resolveModelProgress(
                        mediaId,
                        data.modelName,
                        data.success
                    );
                }
                break;
        }
    }

    // RENAMED: from startVideoProcessing to startMediaProcessing
    startMediaProcessing(mediaId, message) {
        if (this.mediaToastMap.has(mediaId)) return;

        const toastId = showToast.loading(<ToastProgress message={message} />, {
            duration: Infinity,
        });
        this.mediaToastMap.set(mediaId, toastId);
        console.log(`[Orchestrator] ✨ Created main toast (ID: ${toastId}) for media ${mediaId}`);
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
        } else {
            showToast.loading(toastContent, { id: toastId });
        }
    }

    resolveModelProgress(mediaId, modelName, success) {
        const modelToastKey = `${mediaId}-${modelName}`;
        const toastId = this.modelToastMap.get(modelToastKey);

        if (toastId) {
            const message = `Analysis ${success ? "complete" : "failed"}.`;
            const toastFn = success ? showToast.success : showToast.error;

            toastFn(<ToastProgress modelName={modelName} message={message} />, {
                id: toastId,
                duration: 5000,
            });
            this.modelToastMap.delete(modelToastKey);
        }
    }

    // RENAMED: from resolveVideoProcessing to resolveMediaProcessing
    resolveMediaProcessing(mediaId, filename, success, errorMsg = "") {
        const mainToastId = this.mediaToastMap.get(mediaId);
        if (mainToastId) {
            const message = success
                ? `All analyses for "${filename}" are complete!`
                : `Processing failed: ${errorMsg}`;
            const toastFn = success ? showToast.success : showToast.error;

            toastFn(<ToastProgress message={message} />, {
                id: mainToastId,
                duration: 8000,
            });
            this.mediaToastMap.delete(mediaId);
        }
    }
}

export const toastOrchestrator = new ToastOrchestrator();