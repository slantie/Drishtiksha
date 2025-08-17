// src/components/ui/ToastProgress.jsx

import React from "react";
import { Loader2 } from "lucide-react";

// REFACTOR: Enhanced the visual design for better readability and a more modern look inside toasts.
export const ToastProgress = ({ message, modelName, progress, total }) => {
    const percentage =
        total > 0 ? Math.min(100, (progress / total) * 100) : null;

    return (
        <div className="flex items-center space-x-3">
            {/* <div>
                <Loader2 className="h-5 w-5 animate-spin text-primary-main" />
            </div> */}
            <div className="flex-grow">
                {modelName && (
                    <p className="font-semibold text-sm">{modelName}</p>
                )}
                <p
                    className={`text-sm ${
                        modelName ? "text-gray-500 dark:text-gray-400" : ""
                    }`}
                >
                    {message}
                </p>
                {percentage !== null && (
                    <div className="w-full bg-light-muted-background dark:bg-dark-secondary rounded-full h-1.5 mt-2 overflow-hidden">
                        <div
                            className="bg-primary-main h-1.5 rounded-full transition-all duration-300"
                            style={{ width: `${percentage}%` }}
                        ></div>
                    </div>
                )}
            </div>
        </div>
    );
};
