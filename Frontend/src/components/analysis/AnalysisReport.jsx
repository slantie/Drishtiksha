// src/components/analysis/AnalysisReport.jsx

import React from "react";
// Import all our new, modular chart components
import { OverallResultCard } from "./charts/OverallResultCard";
import { EnvironmentCard } from "./charts/EnvironmentCard";
import { VideoReport } from "./charts/video/VideoReport";
import { AudioReport } from "./charts/audio/AudioReport";

export const AnalysisReport = ({ result }) => {
  if (!result) {
    return <p>No analysis data available to display.</p>;
  }

  const isVideo = result.media_type === "video";
  const isAudio = result.media_type === "audio";

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
      {/* Left Column: Consistent for all media types */}
      <div className="lg:col-span-1 space-y-6 sticky top-24">
        <OverallResultCard result={result} />
        {/* The EnvironmentCard can be created by adapting the logic from the old page */}
        {/* <EnvironmentCard result={result} /> */}
      </div>

      {/* Right Column: Dynamically renders the correct report type */}
      <div className="lg:col-span-2 space-y-6">
        {isVideo && <VideoReport result={result} />}
        {isAudio && <AudioReport result={result} />}
      </div>
    </div>
  );
};
