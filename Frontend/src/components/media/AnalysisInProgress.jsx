// src/components/media/AnalysisInProgress.jsx

import React, { useMemo } from "react";
import {
  Loader2,
  Brain,
  CheckCircle,
  Clock,
  AlertTriangle,
  Lightbulb,
  BarChart3,
} from "lucide-react"; // Added BarChart3 icon
import { useMediaProgress } from "../../hooks/useMediaProgess.js";
import { useAnalysisProgress } from "../../hooks/useAnalysisProgress.jsx";
import { useServerStatusQuery } from "../../hooks/useMonitoringQuery.js";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "../ui/Card.jsx";
import { Button } from "../ui/Button.jsx";
import { LoadingSpinner } from "../ui/LoadingSpinner.jsx";
import { AnalysisProgress } from "../ui/AnalysisProgress.jsx";
import { getMediaType } from "../../utils/media.js";

// Enhanced helper component for displaying individual model status with progress
const ModelStatusRow = ({ model, status, progress = null }) => {
  let icon,
    color,
    text,
    progressBar = null;

  switch (status) {
    case "COMPLETED":
      icon = <CheckCircle className="h-4 w-4 text-green-500" />;
      color = "text-green-600";
      text = "Completed";
      break;
    case "PROCESSING":
      icon = <Loader2 className="h-4 w-4 text-yellow-500 animate-spin" />;
      color = "text-yellow-600";
      text = "Processing...";

      // Add progress bar if we have progress data
      if (progress && progress.total > 0) {
        const percentage = (progress.progress / progress.total) * 100;
        text = `Processing... ${percentage.toFixed(1)}%`;
        progressBar = (
          <div className="w-full bg-gray-200 rounded-full h-1.5 mt-1">
            <div
              className="bg-yellow-500 h-1.5 rounded-full transition-all duration-300"
              style={{ width: `${percentage}%` }}
            />
          </div>
        );
      }
      break;
    case "FAILED":
      icon = <AlertTriangle className="h-4 w-4 text-red-500" />;
      color = "text-red-600";
      text = "Failed";
      break;
    case "PENDING":
    default:
      icon = <Clock className="h-4 w-4 text-gray-500" />;
      color = "text-gray-500";
      text = "Queued/Pending";
      break;
  }

  return (
    <div className="flex flex-col p-3 rounded-md border transition-colors">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Brain className="h-5 w-5 text-primary-main" />
          <span className="font-semibold">{model.name}</span>
        </div>
        <div className={`flex items-center gap-2 ${color}`}>
          {icon}
          <span className="text-sm">{text}</span>
        </div>
      </div>
      {progressBar}
      {progress?.details?.phase && (
        <div className="text-xs text-gray-500 mt-1">
          Phase: {progress.details.phase.replace("_", " ")}
        </div>
      )}
    </div>
  );
};

export const AnalysisInProgress = ({ media }) => {
  // Call all hooks unconditionally
  const mediaId = media?.id || "";
  const mediaFilename = media?.filename || "";
  const mediaMimetype = media?.mimetype || "";

  const { latestProgress, progressEvents } = useMediaProgress(mediaId);
  const { data: serverStatus, isLoading: isServerStatusLoading } =
    useServerStatusQuery();

  // Use the enhanced analysis progress hook
  const {
    isProgressVisible,
    modelProgress,
    showProgress,
    hideProgress,
    overallProgress,
    activeModels,
    analysisProgressProps,
  } = useAnalysisProgress(mediaId, mediaFilename);

  const modelsForThisRun = useMemo(() => {
    if (!serverStatus?.models_info || !mediaMimetype) return [];
    const mediaType = getMediaType(mediaMimetype);

    return serverStatus.models_info.filter((model) => {
      if (!model.loaded) return false;
      // Determine compatibility based on mediaType and model's capabilities
      const isVideoOrImage = mediaType === "VIDEO" || mediaType === "IMAGE";
      const isAudio = mediaType === "AUDIO";

      return (isVideoOrImage && model.is_video) || (isAudio && model.is_audio);
    });
  }, [serverStatus, mediaMimetype]);

  const modelStatus = useMemo(() => {
    const statusMap = {};
    modelsForThisRun.forEach((model) => {
      statusMap[model.name] = "PENDING";
    });

    // First check our enhanced progress data
    Object.entries(modelProgress).forEach(([modelName, progress]) => {
      if (Object.prototype.hasOwnProperty.call(statusMap, modelName)) {
        if (progress.phase === "complete") statusMap[modelName] = "COMPLETED";
        else if (progress.phase === "failed") statusMap[modelName] = "FAILED";
        else statusMap[modelName] = "PROCESSING";
      }
    });

    // Fallback to legacy progress events
    progressEvents.forEach((event) => {
      const modelName = event.data?.model_name;
      if (
        modelName &&
        Object.prototype.hasOwnProperty.call(statusMap, modelName)
      ) {
        if (event.event.includes("STARTED"))
          statusMap[modelName] = "PROCESSING";
        if (event.event.includes("PROGRESS"))
          statusMap[modelName] = "PROCESSING";
        if (event.event.includes("COMPLETED"))
          statusMap[modelName] = "COMPLETED";
        if (event.event.includes("FAILED")) statusMap[modelName] = "FAILED";
      }
    });

    return statusMap;
  }, [progressEvents, modelsForThisRun, modelProgress]);

  // Early return after all hooks are called
  if (!media) {
    return (
      <div className="mx-auto text-center max-w-4xl space-y-4">
        <AlertTriangle className="h-12 w-12 text-red-500 mx-auto" />
        <h1 className="text-3xl font-bold">Media Data Unavailable</h1>
        <p className="text-lg text-light-muted-text dark:text-dark-muted-text">
          Cannot display analysis progress without media information.
        </p>
      </div>
    );
  }

  const latestEventMessage =
    latestProgress?.message ||
    `Your ${media.mediaType.toLowerCase()} is being queued for analysis...`;

  const completedModels = Object.values(modelStatus).filter(
    (status) => status === "COMPLETED"
  ).length;
  const failedModels = Object.values(modelStatus).filter(
    (status) => status === "FAILED"
  ).length;
  const processingModels = Object.values(modelStatus).filter(
    (status) => status === "PROCESSING"
  ).length;

  return (
    <>
      <div className="mx-auto text-center max-w-4xl">
        <LoadingSpinner size="md" />
        <h1 className="text-3xl font-bold mt-4">Analysis in Progress</h1>
        <p className="text-lg text-light-muted-text dark:text-dark-muted-text mt-2 mb-4">
          Your file "{media.filename}" is being analyzed. This page will update
          automatically.
        </p>

        {/* Enhanced Progress Summary */}
        {activeModels > 0 && (
          <div className="mb-6 p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
            <div className="flex items-center justify-center gap-4 mb-2">
              <span className="text-sm font-medium">
                Overall Progress: {overallProgress.toFixed(1)}%
              </span>
              <Button
                onClick={showProgress}
                variant="outline"
                size="sm"
                className="gap-2"
              >
                <BarChart3 className="h-4 w-4" />
                View Detailed Progress
              </Button>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${overallProgress}%` }}
              />
            </div>
            <div className="flex justify-center gap-4 mt-2 text-xs text-gray-600">
              <span>✅ {completedModels} Completed</span>
              <span>⚡ {processingModels} Processing</span>
              {failedModels > 0 && <span>❌ {failedModels} Failed</span>}
            </div>
          </div>
        )}

        <Card>
          <CardHeader>
            <CardTitle>
              Current Status:{" "}
              <span className="capitalize text-primary-main">
                {media.status.replace("_", " ").toLowerCase()}
              </span>
            </CardTitle>
            <CardDescription className="flex items-center justify-center gap-2">
              <Lightbulb className="h-4 w-4 text-yellow-500 flex-shrink-0" />
              <span>{latestEventMessage}</span>
            </CardDescription>
          </CardHeader>
          <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {isServerStatusLoading ? (
              <p className="text-light-muted-text dark:text-dark-muted-text text-center col-span-full">
                Loading model status...
              </p>
            ) : modelsForThisRun.length > 0 ? (
              modelsForThisRun.map((model) => (
                <ModelStatusRow
                  key={model.name}
                  model={model}
                  status={modelStatus[model.name]}
                  progress={modelProgress[model.name]}
                />
              ))
            ) : (
              <p className="text-light-muted-text dark:text-dark-muted-text text-center col-span-full">
                No compatible analysis models loaded or available for "
                {media.mediaType.toLowerCase()}" media.
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Enhanced Analysis Progress Modal */}
      <AnalysisProgress {...analysisProgressProps} />
    </>
  );
};
