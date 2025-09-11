// Frontend/src/components/ui/ToastProgress.jsx

import React, { useEffect, useState } from "react";
import { cn } from "../../lib/utils";

/**
 * Enhanced progress component with TQDM-style progress bar and statistics
 * Designed to work both in toasts and standalone components
 */
export const ToastProgress = ({
  message,
  modelName,
  progress,
  total,
  details = {},
}) => {
  const [startTime] = useState(Date.now());
  const [elapsedTime, setElapsedTime] = useState(0);

  const percentage = total > 0 ? Math.min(100, (progress / total) * 100) : null;
  const hasProgress = percentage !== null && !isNaN(percentage);

  // Update elapsed time every second
  useEffect(() => {
    const interval = setInterval(() => {
      setElapsedTime(Date.now() - startTime);
    }, 1000);

    return () => clearInterval(interval);
  }, [startTime]);

  // Calculate statistics similar to TQDM
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

  const calculateETA = () => {
    if (!hasProgress || progress === 0 || percentage >= 100) return null;

    const avgTimePerItem = elapsedTime / progress;
    const remainingItems = total - progress;
    const etaMs = avgTimePerItem * remainingItems;

    return formatTime(etaMs);
  };

  const calculateSpeed = () => {
    if (!hasProgress || elapsedTime === 0 || progress === 0) return null;

    const itemsPerSecond = (progress * 1000) / elapsedTime;
    return itemsPerSecond.toFixed(1);
  };

  const eta = calculateETA();
  const speed = calculateSpeed();

  return (
    <div className="flex items-start space-x-3 text-light-text dark:text-dark-text min-w-0 flex-grow">
      <div className="flex-grow min-w-0">
        {/* Header with model name and elapsed time */}
        {modelName && (
          <div className="flex items-center justify-between mb-1">
            <p className="font-bold text-sm opacity-90 truncate">{modelName}</p>
            <span className="text-xs opacity-75 ml-2 flex-shrink-0">
              {formatTime(elapsedTime)}
            </span>
          </div>
        )}

        {/* Main message */}
        <p
          className={cn(
            "text-sm mb-2 break-words",
            modelName ? "opacity-80" : "font-semibold"
          )}
        >
          {message}
        </p>

        {/* Progress bar with TQDM-style visualization */}
        {hasProgress && (
          <div className="space-y-1">
            {/* Progress statistics */}
            <div className="flex items-center justify-between text-xs opacity-75">
              <span className="truncate">
                {progress}/{total} ({percentage.toFixed(1)}%)
              </span>
              <div className="flex items-center space-x-2 ml-2 flex-shrink-0">
                {speed && <span>{speed} it/s</span>}
                {eta && <span>ETA: {eta}</span>}
              </div>
            </div>

            {/* Enhanced progress bar */}
            <div className="relative">
              <div className="w-full bg-white/20 rounded-full h-2 overflow-hidden">
                <div
                  className={cn(
                    "h-2 rounded-full transition-all duration-300 relative",
                    percentage < 100
                      ? "bg-gradient-to-r from-blue-400 to-blue-600"
                      : "bg-gradient-to-r from-green-400 to-green-600"
                  )}
                  style={{ width: `${percentage}%` }}
                >
                  {/* Animated shine effect for active progress */}
                  {percentage < 100 && percentage > 0 && (
                    <div
                      className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent"
                      style={{
                        animation: "shine 2s ease-in-out infinite",
                      }}
                    />
                  )}
                </div>
              </div>

              {/* TQDM-style character progress bar for visual appeal */}
              <div className="flex mt-1 text-xs font-mono opacity-60">
                <span className="text-blue-300">
                  {Array(Math.floor(percentage / 5))
                    .fill("█")
                    .join("")}
                </span>
                <span className="text-white/30">
                  {Array(20 - Math.floor(percentage / 5))
                    .fill("░")
                    .join("")}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Additional details for specific phases */}
        {(details?.faces_detected_so_far ||
          details?.current_frame_faces ||
          details?.windows_processed) && (
          <div className="text-xs opacity-75 mt-1 space-y-0.5">
            {details.faces_detected_so_far && (
              <div>Total faces detected: {details.faces_detected_so_far}</div>
            )}
            {details.current_frame_faces && (
              <div>Faces in current frame: {details.current_frame_faces}</div>
            )}
            {details.windows_processed && (
              <div>Windows analyzed: {details.windows_processed}</div>
            )}
          </div>
        )}

        {/* Indeterminate progress for phases without specific progress */}
        {!hasProgress && details?.phase && (
          <div className="w-full bg-white/20 rounded-full h-1.5 overflow-hidden mt-2">
            <div
              className="h-1.5 bg-gradient-to-r from-blue-400 to-blue-600 rounded-full"
              style={{
                width: "40%",
                animation: "slide 1.5s ease-in-out infinite alternate",
              }}
            />
          </div>
        )}

        <style jsx>{`
          @keyframes shine {
            0% {
              transform: translateX(-100%);
            }
            100% {
              transform: translateX(100%);
            }
          }

          @keyframes slide {
            0% {
              margin-left: 0%;
            }
            100% {
              margin-left: 60%;
            }
          }
        `}</style>
      </div>
    </div>
  );
};
