// src/components/ui/SkeletonCard.jsx

import React from "react";
import { Card } from "./Card";
import { cn } from "../../lib/utils";

// REFACTOR: Simplified and aligned with the theme's colors for a more subtle loading state.
export const SkeletonCard = ({ className }) => {
    return (
        <Card className={cn("animate-pulse", className)}>
            <div className="p-6 space-y-4">
                <div className="h-6 w-2/3 rounded bg-light-hover dark:bg-dark-hover"></div>
                <div className="space-y-2">
                    <div className="h-4 w-full rounded bg-light-hover dark:bg-dark-hover"></div>
                    <div className="h-4 w-5/6 rounded bg-light-hover dark:bg-dark-hover"></div>
                </div>
            </div>
        </Card>
    );
};
