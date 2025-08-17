// src/lib/toastOrchestrator.jsx

import React from "react";
import { showToast } from "../utils/toast";
import { ToastProgress } from "../components/ui/ToastProgress.jsx";
import { CheckCircle, AlertTriangle } from "lucide-react";

class ToastOrchestrator {
    videoToastMap = new Map();
    modelToastMap = new Map();

    startVideoProcessing(videoId, message) {
        if (this.videoToastMap.has(videoId)) return;

        const toastId = showToast.loading(<ToastProgress message={message} />, {
            duration: Infinity,
        });
        this.videoToastMap.set(videoId, toastId);
        console.log(
            `[Orchestrator] ✨ Created main toast (ID: ${toastId}) for video ${videoId}`
        );
    }

    updateModelProgress(videoId, modelName, message, progress, total) {
        const modelToastKey = `${videoId}-${modelName}`;
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
                `[Orchestrator] ✨ Created model toast (ID: ${toastId}) for ${modelName}`
            );
        } else {
            showToast.loading(toastContent, { id: toastId });
        }
    }

    resolveModelProgress(videoId, modelName, success) {
        const modelToastKey = `${videoId}-${modelName}`;
        const toastId = this.modelToastMap.get(modelToastKey);

        if (toastId) {
            const message = `Analysis ${success ? "complete" : "failed"}.`;
            const toastFn = success ? showToast.success : showToast.error;

            toastFn(<ToastProgress modelName={modelName} message={message} />, {
                id: toastId,
                duration: 5000,
            });
            this.modelToastMap.delete(modelToastKey);
            console.log(
                `[Orchestrator] ✅ Resolved model toast (ID: ${toastId}) for ${modelName}`
            );
        }
    }

    resolveVideoProcessing(videoId, filename, success, errorMsg = "") {
        const mainToastId = this.videoToastMap.get(videoId);
        if (mainToastId) {
            const message = success
                ? `All analyses for "${filename}" are complete!`
                : `Processing failed: ${errorMsg}`;
            const toastFn = success ? showToast.success : showToast.error;

            toastFn(<ToastProgress message={message} />, {
                id: mainToastId,
                duration: 8000,
            });
            this.videoToastMap.delete(videoId);
            console.log(
                `[Orchestrator] ✅ Resolved main video toast (ID: ${mainToastId})`
            );
        }
    }
}

export const toastOrchestrator = new ToastOrchestrator();
