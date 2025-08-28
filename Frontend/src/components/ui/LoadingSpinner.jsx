// src/components/ui/LoadingSpinner.jsx

import React from "react";
import { Loader2 } from "lucide-react";
import { cn } from "../../lib/utils";

// REFACTOR: Simplified to a single, consistent spinner for a unified brand feel.
// Removed multiple variants (dots, pulse, etc.) in favor of one clean, themeable spinner.
const sizeClasses = {
    sm: "h-4 w-4",
    md: "h-6 w-6",
    lg: "h-10 w-10",
};

export const DotsSpinner =({
size = "md", color="text-light-highlight dark:text-dark-highlight"
}) => {
    const dotSize =
        size === "xs"
            ? "w-1 h-1"
            : size === "sm"
            ? "w-1.5 h-1.5"
            : size === "md"
            ? "w-2 h-2"
            : size === "lg"
            ? "w-2.5 h-2.5"
            : "w-3 h-3";

    return (
        <div className="flex space-x-1">
            <div
                className={`${dotSize} ${color} bg-current rounded-full animate-bounce [animation-delay:0ms]`}
            />
            <div
                className={`${dotSize} ${color} bg-current rounded-full animate-bounce [animation-delay:150ms]`}
            />
            <div
                className={`${dotSize} ${color} bg-current rounded-full animate-bounce [animation-delay:300ms]`}
            />
        </div>
    );
};

export const LoadingSpinner = ({ size = "md", text, className }) => (
    <div
        className={cn(
            "flex flex-col items-center justify-center gap-2",
            className
        )}
    >
        <Loader2
            className={cn("animate-spin text-primary-main", sizeClasses[size])}
        />
        {text && (
            <p className="text-sm text-light-muted-text dark:text-dark-muted-text">
                {text}
            </p>
        )}
    </div>
);

export const PageLoader = ({ text = "Loading..." }) => (
    <div className="flex h-screen w-full items-center justify-center">
        <LoadingSpinner size="lg" text={text} />
    </div>
);
