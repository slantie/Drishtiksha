// src/components/media/AnalysisInProgress.jsx

import React, { useMemo } from "react";
import {
  Loader2,
  Brain,
  CheckCircle,
  Clock,
  AlertTriangle,
  Lightbulb,
} from "lucide-react"; // Added Lightbulb icon
import { useMediaProgress } from "../../hooks/useMediaProgess.js";
import { useServerStatusQuery } from "../../hooks/useMonitoringQuery.js";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "../ui/Card.jsx";
import { DotsSpinner } from "../ui/LoadingSpinner.jsx";
import { getMediaType } from "../../utils/media.js";

// Helper component for displaying individual model status
const ModelStatusRow = ({ model, status }) => {
  let icon, color, text;
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
    <div className="flex items-center justify-between p-2 rounded-md transition-colors">
      <div className="flex items-center gap-3">
        <Brain className="h-5 w-5 text-primary-main" />
        <span className="font-semibold">{model.name}</span>
      </div>
      <div className={`flex items-center gap-2 ${color}`}>
        {icon}
        <span className="text-sm">{text}</span>
      </div>
    </div>
  );
};

export const AnalysisInProgress = ({ media }) => {
  // Ensure media object is available
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

  const { latestProgress, progressEvents } = useMediaProgress(media.id);
  const { data: serverStatus, isLoading: isServerStatusLoading } =
    useServerStatusQuery();

  const modelsForThisRun = useMemo(() => {
    if (!serverStatus?.models_info) return [];
    const mediaType = getMediaType(media.mimetype);

    return serverStatus.models_info.filter((model) => {
      if (!model.loaded) return false;
      // Determine compatibility based on mediaType and model's capabilities
      const isVideoOrImage = mediaType === "VIDEO" || mediaType === "IMAGE";
      const isAudio = mediaType === "AUDIO";

      return (isVideoOrImage && model.is_video) || (isAudio && model.is_audio);
    });
  }, [serverStatus, media.mimetype]);

  const modelStatus = useMemo(() => {
    const statusMap = {};
    modelsForThisRun.forEach((model) => {
      statusMap[model.name] = "PENDING";
    });

    // Iterate through all progress events to determine the latest status for each model
    progressEvents.forEach((event) => {
      const modelName = event.data?.model_name; // Use model_name from backend payload
      if (modelName && statusMap.hasOwnProperty(modelName)) {
        // Check if the model is one we're tracking
        if (event.event.includes("STARTED"))
          statusMap[modelName] = "PROCESSING";
        if (event.event.includes("PROGRESS"))
          statusMap[modelName] = "PROCESSING"; // Frame progress also means processing
        if (event.event.includes("COMPLETED"))
          statusMap[modelName] = "COMPLETED";
        if (event.event.includes("FAILED")) statusMap[modelName] = "FAILED";
      }
    });
    return statusMap;
  }, [progressEvents, modelsForThisRun]);

  const latestEventMessage =
    latestProgress?.message ||
    `Your ${media.mediaType.toLowerCase()} is being queued for analysis...`;

  return (
    <div className="mx-auto text-center max-w-4xl">
      <DotsSpinner size="lg" />
      <h1 className="text-3xl font-bold mt-4">Analysis in Progress</h1>
      <p className="text-lg text-light-muted-text dark:text-dark-muted-text mt-2 mb-8">
        Your file "{media.filename}" is being analyzed. This page will update
        automatically.
      </p>
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
        <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-2">
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
  );
};
