// src/components/ui/ToastProgress.jsx

import React from "react";
import { cn } from "../../lib/utils";
import { Loader2, BrainCircuit } from "lucide-react"; // Reintroduced Loader2, added BrainCircuit for model-specific context

/**
 * Displays progress information within a toast notification.
 * @param {object} props - The component props.
 * @param {string} props.message - The main message to display.
 * @param {string} [props.modelName] - Optional name of the model performing the task.
 * @param {number} [props.progress] - Current progress value.
 * @param {number} [props.total] - Total value for progress calculation.
 * @param {boolean} [props.isError=false] - If true, indicates an error state for styling.
 */
export const ToastProgress = ({
  message,
  modelName,
  progress,
  total,
  isError = false,
}) => {
  const percentage = total > 0 ? Math.min(100, (progress / total) * 100) : null;

  return (
    <div className="flex items-center space-x-3">
      {/* Display a rotating loader, or a static brain if it's a model toast without specific progress */}
      <div>
        {modelName ? (
          <BrainCircuit
            className={cn("h-5 w-5", !isError && "text-primary-main")}
          /> // Consistent model icon
        ) : (
          <Loader2
            className={cn(
              "h-5 w-5 animate-spin",
              !isError && "text-primary-main"
            )}
          />
        )}
      </div>
      <div className="flex-grow">
        {modelName && (
          <p
            className={cn(
              "font-semibold text-sm",
              isError ? "text-red-400" : "text-gray-900 dark:text-gray-50"
            )}
          >
            {modelName}
          </p>
        )}
        <p
          className={cn(
            "text-sm",
            modelName
              ? isError
                ? "text-red-300"
                : "text-gray-600 dark:text-gray-300"
              : isError
              ? "text-red-400"
              : "text-gray-900 dark:text-gray-50"
          )}
        >
          {message}
        </p>
        {percentage !== null && (
          <div className="w-full bg-light-muted-background dark:bg-dark-secondary rounded-full h-1.5 mt-2 overflow-hidden">
            <div
              className={cn(
                isError ? "bg-red-500" : "bg-primary-main",
                "h-1.5 rounded-full transition-all duration-300"
              )}
              style={{ width: `${percentage}%` }}
            ></div>
          </div>
        )}
      </div>
    </div>
  );
};
