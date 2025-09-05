// src/components/analysis/AnalysisReport.jsx

import React from "react";
// Import all our new, modular chart components
import { OverallResultCard } from "./charts/OverallResultCard";
import { EnvironmentCard } from "./charts/EnvironmentCard";
import { VideoReport } from "./charts/video/VideoReport";
import { AudioReport } from "./charts/audio/AudioReport";

export const AnalysisReport = ({ result }) => {
  // `result` here IS the analysis.resultPayload directly
  if (!result || Object.keys(result).length === 0) {
    return (
      <div className="text-center p-8 text-light-muted-text dark:text-dark-muted-text">
        <p>No detailed analysis data available to display for this model.</p>
        <p className="mt-2 text-sm">
          This may indicate a failed analysis or an issue with the ML server
          response.
        </p>
      </div>
    );
  }

  const mediaType = result.media_type; // Expecting 'video', 'audio', 'image' from backend resultPayload

  // Data for OverallResultCard
  const overallData = {
    prediction: result.prediction, // Top-level prediction from ML service
    confidence: result.confidence, // Top-level confidence from ML service
  };

  // Data for EnvironmentCard
  const environmentData = {
    modelName: result.model_name, // e.g., SIGLIP-LSTM-V4
    systemInfo: result.system_info, // Contains device_info, python_version, etc.
  };

  const isVideo = mediaType === "video";
  const isAudio = mediaType === "audio";

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
      {/* Left Column: Consistent for all media types */}
      <div className="lg:col-span-1 space-y-6 lg:sticky lg:top-24">
        <OverallResultCard result={overallData} />
        <EnvironmentCard result={environmentData} /> {/* Reintegrated */}
      </div>

      {/* Right Column: Dynamically renders the correct report type */}
      <div className="lg:col-span-2 space-y-6">
        {isVideo && <VideoReport result={result} />}
        {isAudio && <AudioReport result={result} />}
        {/* Potentially add ImageReport here if needed later */}
      </div>
    </div>
  );
};
