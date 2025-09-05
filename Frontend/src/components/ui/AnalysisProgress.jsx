// Frontend/src/components/ui/AnalysisProgress.jsx

import React, { useState, useEffect } from "react";
import { cn } from "../../lib/utils";
import { Card } from "./Card";

/**
 * Comprehensive analysis progress component showing multiple models progress
 * Similar to TQDM but for multiple concurrent processes
 */
export const AnalysisProgress = ({
  mediaId,
  filename,
  isVisible = false,
  onClose,
  progressData = {},
}) => {
  const [startTime] = useState(Date.now());
  const [elapsedTime, setElapsedTime] = useState(0);
  const [modelProgress, setModelProgress] = useState({});

  useEffect(() => {
    const interval = setInterval(() => {
      setElapsedTime(Date.now() - startTime);
    }, 1000);

    return () => clearInterval(interval);
  }, [startTime]);

  // Update model progress when new data comes in
  useEffect(() => {
    if (progressData.modelName && progressData.progress !== undefined) {
      setModelProgress((prev) => ({
        ...prev,
        [progressData.modelName]: {
          ...progressData,
          lastUpdate: Date.now(),
        },
      }));
    }
  }, [progressData]);

  const formatTime = (ms) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) {
      return `${hours}:${String(minutes % 60).padStart(2, "0")}:${String(
        seconds % 60
      ).padStart(2, "0")}`;
    } else if (minutes > 0) {
      return `${minutes}:${String(seconds % 60).padStart(2, "0")}`;
    } else {
      return `${seconds}s`;
    }
  };

  const calculateOverallProgress = () => {
    const models = Object.values(modelProgress);
    if (models.length === 0) return 0;

    const totalProgress = models.reduce((sum, model) => {
      const percentage =
        model.total > 0 ? (model.progress / model.total) * 100 : 0;
      return sum + percentage;
    }, 0);

    return totalProgress / models.length;
  };

  const ModelProgressBar = ({ modelName, data }) => {
    const percentage =
      data.total > 0 ? Math.min(100, (data.progress / data.total) * 100) : 0;
    const timeSinceUpdate = Date.now() - data.lastUpdate;
    const isStale = timeSinceUpdate > 10000; // 10 seconds

    const calculateETA = () => {
      if (!data.progress || data.progress === 0 || percentage >= 100)
        return null;

      const timeElapsed = data.lastUpdate - startTime;
      const avgTimePerItem = timeElapsed / data.progress;
      const remainingItems = data.total - data.progress;
      const etaMs = avgTimePerItem * remainingItems;

      return formatTime(etaMs);
    };

    const calculateSpeed = () => {
      if (!data.progress || data.progress === 0) return null;

      const timeElapsed = data.lastUpdate - startTime;
      if (timeElapsed === 0) return null;

      const itemsPerSecond = (data.progress * 1000) / timeElapsed;
      return itemsPerSecond.toFixed(1);
    };

    return (
      <div className={cn("p-3 border rounded-lg", isStale ? "opacity-60" : "")}>
        <div className="flex items-center justify-between mb-2">
          <span className="font-medium text-sm">{modelName}</span>
          <div className="flex items-center space-x-2 text-xs text-muted-foreground">
            {calculateSpeed() && <span>{calculateSpeed()} it/s</span>}
            {calculateETA() && <span>ETA: {calculateETA()}</span>}
          </div>
        </div>

        <div className="space-y-1">
          <div className="flex justify-between text-xs">
            <span>
              {data.progress || 0}/{data.total || 0}
            </span>
            <span>{percentage.toFixed(1)}%</span>
          </div>

          <div className="w-full bg-muted rounded-full h-2">
            <div
              className={cn(
                "h-2 rounded-full transition-all duration-300",
                percentage >= 100
                  ? "bg-green-500"
                  : isStale
                  ? "bg-yellow-500"
                  : "bg-blue-500"
              )}
              style={{ width: `${percentage}%` }}
            />
          </div>

          {/* TQDM-style bar */}
          <div className="flex text-xs font-mono">
            <span className="text-blue-500">
              {Array(Math.floor(percentage / 5))
                .fill("█")
                .join("")}
            </span>
            <span className="text-muted-foreground">
              {Array(20 - Math.floor(percentage / 5))
                .fill("░")
                .join("")}
            </span>
          </div>

          <div className="text-xs text-muted-foreground truncate">
            {data.message || `Processing ${modelName}...`}
          </div>
        </div>
      </div>
    );
  };

  const overallProgress = calculateOverallProgress();
  const activeModels = Object.keys(modelProgress).length;

  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <Card className="w-full max-w-2xl max-h-[80vh] overflow-y-auto">
        <div className="p-6 space-y-4">
          {/* Header */}
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold">Analysis Progress</h3>
              <p className="text-sm text-muted-foreground">{filename}</p>
            </div>
            <div className="flex items-center space-x-4">
              <span className="text-sm text-muted-foreground">
                {formatTime(elapsedTime)}
              </span>
              <button
                onClick={onClose}
                className="text-muted-foreground hover:text-foreground"
              >
                ✕
              </button>
            </div>
          </div>

          {/* Overall Progress */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Overall Progress</span>
              <span>
                {overallProgress.toFixed(1)}% ({activeModels} models)
              </span>
            </div>
            <div className="w-full bg-muted rounded-full h-3">
              <div
                className="h-3 bg-gradient-to-r from-blue-500 to-green-500 rounded-full transition-all duration-300"
                style={{ width: `${overallProgress}%` }}
              />
            </div>

            {/* Overall TQDM-style visualization */}
            <div className="flex text-sm font-mono justify-center">
              <span className="text-blue-500">
                {Array(Math.floor(overallProgress / 2.5))
                  .fill("█")
                  .join("")}
              </span>
              <span className="text-muted-foreground">
                {Array(40 - Math.floor(overallProgress / 2.5))
                  .fill("░")
                  .join("")}
              </span>
            </div>
          </div>

          {/* Individual Model Progress */}
          <div className="space-y-3">
            <h4 className="font-medium">Model Progress</h4>
            {Object.entries(modelProgress).map(([modelName, data]) => (
              <ModelProgressBar
                key={modelName}
                modelName={modelName}
                data={data}
              />
            ))}

            {activeModels === 0 && (
              <div className="text-center py-8 text-muted-foreground">
                <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-2" />
                Initializing analysis...
              </div>
            )}
          </div>

          {/* Summary Stats */}
          <div className="pt-4 border-t text-xs text-muted-foreground space-y-1">
            <div>Media ID: {mediaId}</div>
            <div>Started: {new Date(startTime).toLocaleTimeString()}</div>
            <div>Active Models: {activeModels}</div>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default AnalysisProgress;
