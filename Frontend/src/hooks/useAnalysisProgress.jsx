// Frontend/src/hooks/useAnalysisProgress.jsx

import { useState, useEffect, useCallback } from "react";
import { toastOrchestrator } from "../lib/toastOrchestrator";

/**
 * Hook to manage analysis progress tracking for a media item
 * Provides detailed progress data and modal control
 */
export const useAnalysisProgress = (mediaId, filename) => {
  const [isProgressVisible, setIsProgressVisible] = useState(false);
  const [progressData, setProgressData] = useState({});
  const [modelProgress, setModelProgress] = useState({});

  // Handle progress events from the toast orchestrator
  const handleProgressUpdate = useCallback(
    (event) => {
      const { event: eventType, message, data } = event;

      setProgressData((prev) => ({
        ...prev,
        lastEvent: eventType,
        lastMessage: message,
        lastData: data,
        timestamp: Date.now(),
      }));

      // Update model-specific progress
      if (data?.model_name) {
        setModelProgress((prev) => {
          const phase = 
            eventType === "ANALYSIS_STARTED" ? "analyzing" :
            eventType === "ANALYSIS_COMPLETED" ? "completed" :
            eventType === "ANALYSIS_FAILED" ? "failed" :
            data.phase || prev[data.model_name]?.phase || "processing";

          return {
            ...prev,
            [data.model_name]: {
              modelName: data.model_name,
              message: message,
              progress: data.progress,
              total: data.total,
              details: data.details,
              timestamp: Date.now(),
              phase: phase,
              prediction: data.prediction,
              confidence: data.confidence,
              error: data.error_message,
            },
          };
        });
      }

      // Auto-show progress modal for certain events
      if (["PROCESSING_STARTED", "ANALYSIS_STARTED"].includes(eventType)) {
        setIsProgressVisible(true);
      }

      // Auto-hide when all processing is complete or failed
      if (["ANALYSIS_COMPLETE", "ANALYSIS_FAILED"].includes(eventType)) {
        // Use setTimeout to allow final updates to render
        setTimeout(() => {
          setModelProgress((currentProgress) => {
            const models = Object.values(currentProgress);
            const allDone = models.length > 0 && 
              models.every((model) => 
                model.phase === "completed" || model.phase === "failed"
              );
            
            if (allDone) {
              // Auto-hide after 3 seconds if all models are done
              setTimeout(() => setIsProgressVisible(false), 3000);
            }
            
            return currentProgress;
          });
        }, 100);
      }
    },
    [modelProgress]
  );

  // Listen for custom events to show progress modal
  useEffect(() => {
    const handleShowProgress = (event) => {
      if (event.detail.mediaId === mediaId) {
        setIsProgressVisible(true);
      }
    };

    window.addEventListener("showAnalysisProgress", handleShowProgress);
    return () =>
      window.removeEventListener("showAnalysisProgress", handleShowProgress);
  }, [mediaId]);

  // Register/unregister progress callback
  useEffect(() => {
    if (mediaId) {
      toastOrchestrator.registerProgressCallback(mediaId, handleProgressUpdate);

      return () => {
        toastOrchestrator.unregisterProgressCallback(mediaId);
      };
    }
  }, [mediaId, handleProgressUpdate]);

  const showProgress = () => setIsProgressVisible(true);
  const hideProgress = () => setIsProgressVisible(false);

  const getOverallProgress = () => {
    const models = Object.values(modelProgress);
    if (models.length === 0) return 0;

    const totalProgress = models.reduce((sum, model) => {
      const percentage =
        model.total > 0 ? (model.progress / model.total) * 100 : 0;
      return sum + percentage;
    }, 0);

    return totalProgress / models.length;
  };

  const getActiveModels = () => Object.keys(modelProgress).length;

  const isAnalysisComplete = () => {
    const models = Object.values(modelProgress);
    return (
      models.length > 0 &&
      models.every(
        (model) => model.phase === "complete" || model.phase === "failed"
      )
    );
  };

  const hasFailedModels = () => {
    return Object.values(modelProgress).some(
      (model) => model.phase === "failed"
    );
  };

  return {
    // State
    isProgressVisible,
    progressData,
    modelProgress,

    // Actions
    showProgress,
    hideProgress,

    // Computed values
    overallProgress: getOverallProgress(),
    activeModels: getActiveModels(),
    isComplete: isAnalysisComplete(),
    hasFailed: hasFailedModels(),

    // For component props
    analysisProgressProps: {
      mediaId,
      filename,
      isVisible: isProgressVisible,
      onClose: hideProgress,
      progressData: progressData,
    },
  };
};

export default useAnalysisProgress;
