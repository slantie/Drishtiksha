// Frontend/src/components/ui/AnalysisProgress.jsx

import React, { useState, useEffect, useMemo } from "react";
import { cn } from "../../lib/utils";
import { Card } from "./Card";

/**
 * Comprehensive analysis progress component showing multiple models progress
 * Similar to TQDM but for multiple concurrent processes
 * 
 * FIXED: Now accepts modelProgress directly from useAnalysisProgress hook
 */
export const AnalysisProgress = ({
  mediaId,
  filename,
  isVisible = false,
  onClose,
  progressData = {}, // Legacy format (single model update)
  modelProgress: externalModelProgress = {}, // NEW: Direct model progress map
}) => {
  const [startTime] = useState(Date.now());
  const [elapsedTime, setElapsedTime] = useState(0);
  const [internalModelProgress, setInternalModelProgress] = useState({});

  useEffect(() => {
    const interval = setInterval(() => {
      setElapsedTime(Date.now() - startTime);
    }, 1000);

    return () => clearInterval(interval);
  }, [startTime]);

  // Update model progress when new data comes in (legacy single-model format)
  useEffect(() => {
    if (progressData.modelName && progressData.progress !== undefined) {
      setInternalModelProgress((prev) => ({
        ...prev,
        [progressData.modelName]: {
          ...progressData,
          lastUpdate: Date.now(),
        },
      }));
    }
  }, [progressData]);

  // Merge external model progress (from useAnalysisProgress) with internal
  const modelProgress = useMemo(() => {
    // If external progress exists, use it (it's more complete)
    if (Object.keys(externalModelProgress).length > 0) {
      // Transform to the expected format with lastUpdate
      return Object.entries(externalModelProgress).reduce((acc, [modelName, data]) => {
        // Filter out non-model entries (like "Backend worker", empty names, etc.)
        if (!modelName || 
            modelName.toLowerCase().includes('worker') || 
            modelName.toLowerCase().includes('backend') ||
            modelName.trim() === '') {
          return acc; // Skip this entry
        }
        
        acc[modelName] = {
          modelName: modelName,
          progress: data.progress || 0,
          total: data.total || 100,
          message: data.message || '',
          lastUpdate: data.timestamp || Date.now(),
          phase: data.phase,
          details: data.details,
        };
        return acc;
      }, {});
    }
    // Fallback to internal (legacy)
    return internalModelProgress;
  }, [externalModelProgress, internalModelProgress]);

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
    // Handle phase-based completion
    const isQueued = data.phase === 'queued';
    const isCompleted = data.phase === 'completed' || data.phase === 'complete';
    const isFailed = data.phase === 'failed';
    const isProcessing = data.phase === 'analyzing' || data.phase === 'processing';
    
    const percentage = isCompleted ? 100 :
      isFailed ? 0 :
      isQueued ? 0 :
      data.total > 0 ? Math.min(100, (data.progress / data.total) * 100) : 0;
    
    const timeSinceUpdate = Date.now() - (data.lastUpdate || Date.now());
    const isStale = timeSinceUpdate > 10000 && !isCompleted && !isFailed; // 10 seconds

    const calculateETA = () => {
      if (!data.progress || data.progress === 0 || percentage >= 100 || isCompleted)
        return null;

      const timeElapsed = data.lastUpdate - startTime;
      if (timeElapsed <= 0) return null;
      
      const avgTimePerItem = timeElapsed / data.progress;
      const remainingItems = data.total - data.progress;
      const etaMs = avgTimePerItem * remainingItems;

      return formatTime(etaMs);
    };

    const calculateSpeed = () => {
      if (!data.progress || data.progress === 0 || isCompleted) return null;

      const timeElapsed = data.lastUpdate - startTime;
      if (timeElapsed <= 0) return null;

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

          <div className="w-full bg-light-secondary dark:bg-dark-secondary rounded-full h-2.5">
            <div
              className={cn(
                "h-2.5 rounded-full transition-all duration-500 ease-out",
                isCompleted
                  ? "bg-green-500"
                  : isFailed
                  ? "bg-red-500"
                  : isQueued
                  ? "bg-gray-400"
                  : isStale
                  ? "bg-yellow-500"
                  : "bg-primary-main" // Purple theme color
              )}
              style={{ width: `${percentage}%` }}
            />
          </div>

          <div className="text-xs text-light-muted-text dark:text-dark-muted-text truncate mt-1">
            {isCompleted ? "✅ Analysis complete" :
             isFailed ? "❌ Analysis failed" :
             isQueued ? "⏳ Queued for analysis..." :
             data.message || `Processing ${modelName}...`}
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
            <div className="flex justify-between text-sm text-light-text dark:text-dark-text">
              <span className="font-medium">Overall Progress</span>
              <span>
                {overallProgress.toFixed(1)}% ({activeModels} model{activeModels !== 1 ? 's' : ''})
              </span>
            </div>
            <div className="w-full bg-light-secondary dark:bg-dark-secondary rounded-full h-3">
              <div
                className="h-3 bg-gradient-to-r from-primary-main to-purple-600 rounded-full transition-all duration-500 ease-out shadow-sm"
                style={{ width: `${overallProgress}%` }}
              />
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
