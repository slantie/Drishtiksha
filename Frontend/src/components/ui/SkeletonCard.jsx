// src/components/ui/SkeletonCard.jsx

import React from "react";
import { Card } from "./Card";

/**
 * A reusable skeleton card component for displaying loading states.
 * It uses a pulsing animation and can be customized with a specific height/width via className.
 * @param {{className?: string}} props
 */
export const SkeletonCard = ({ className = "" }) => {
    return (
        <Card className={`bg-gray-100 dark:bg-gray-800/50 ${className}`}>
            <div className="p-6 h-full">
                <div className="space-y-4 h-full animate-pulse">
                    <div className="w-2/3 h-6 bg-gray-200 dark:bg-gray-700 rounded-md"></div>
                    <div className="space-y-2">
                        <div className="w-full h-4 bg-gray-200 dark:bg-gray-700 rounded-md"></div>
                        <div className="w-5/6 h-4 bg-gray-200 dark:bg-gray-700 rounded-md"></div>
                        <div className="w-3/4 h-4 bg-gray-200 dark:bg-gray-700 rounded-md"></div>
                    </div>
                </div>
            </div>
        </Card>
    );
};
