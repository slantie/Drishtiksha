// src/components/videos/AnalysisInProgress.jsx

import React from "react";
import {
    Loader2,
    Brain,
    CheckCircle,
    Clock,
    AlertTriangle,
} from "lucide-react";
import { useVideoProgress } from "../../hooks/useVideoProgess.js";
import {
    Card,
    CardContent,
    CardHeader,
    CardTitle,
    CardDescription,
} from "../ui/Card";
import { MODEL_INFO } from "../../constants/apiEndpoints.js";

// REFACTOR: A small, clean component for displaying the status of a single model.
const ModelStatusRow = ({ model, status }) => {
    const statusIcons = {
        PENDING: <Clock className="w-5 h-5 text-gray-400" title="Pending" />,
        PROCESSING: (
            <Loader2
                className="w-5 h-5 animate-spin text-yellow-500"
                title="Processing"
            />
        ),
        COMPLETED: (
            <CheckCircle className="w-5 h-5 text-green-500" title="Completed" />
        ),
        FAILED: (
            <AlertTriangle className="w-5 h-5 text-red-500" title="Failed" />
        ),
    };

    return (
        <div className="flex items-center justify-between p-4 bg-light-muted-background dark:bg-dark-secondary rounded-lg">
            <div className="flex items-center gap-4">
                <div className="flex-shrink-0 w-10 h-10 bg-primary-main/10 rounded-full flex items-center justify-center">
                    <Brain className="w-5 h-5 text-primary-main" />
                </div>
                <div>
                    <p className="font-semibold">{model.label}</p>
                    <p className="text-xs text-light-muted-text dark:text-dark-muted-text">
                        {model.description}
                    </p>
                </div>
            </div>
            {statusIcons[status]}
        </div>
    );
};

export const AnalysisInProgress = ({ video }) => {
    const { latestProgress, progressEvents } = useVideoProgress(video.id);

    // REFACTOR: Logic for tracking model status is preserved.
    const modelStatus = React.useMemo(() => {
        const statusMap = {};
        Object.keys(MODEL_INFO).forEach((modelName) => {
            statusMap[modelName] = "PENDING";
        });

        progressEvents.forEach((event) => {
            const modelKey = Object.keys(MODEL_INFO).find((key) =>
                event.data?.modelName?.includes(key)
            );
            if (modelKey) {
                if (event.event === "ANALYSIS_STARTED")
                    statusMap[modelKey] = "PROCESSING";
                else if (event.event === "ANALYSIS_COMPLETED")
                    statusMap[modelKey] = event.data.success
                        ? "COMPLETED"
                        : "FAILED";
            }
        });
        return statusMap;
    }, [progressEvents]);

    const latestEventMessage =
        latestProgress?.message || "Your video has been queued for analysis...";
    const formattedStatus = video.status.replace("_", " ").toLowerCase();

    return (
        // REFACTOR: The entire layout is redesigned for a cleaner, more structured appearance.
        <div className="max-w-3xl mx-auto text-center">
            <Loader2 className="w-16 h-16 text-primary-main mx-auto animate-spin mb-6" />
            <h1 className="text-3xl font-bold mb-2">Analysis in Progress</h1>
            <p className="text-lg text-light-muted-text dark:text-dark-muted-text mb-8">
                Your video "{video.filename}" is being analyzed. This page will
                update automatically.
            </p>
            <Card>
                <CardHeader>
                    <CardTitle>
                        Current Status:{" "}
                        <span className="capitalize text-primary-main">
                            {formattedStatus}
                        </span>
                    </CardTitle>
                    <CardDescription>{latestEventMessage}</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                    {Object.entries(MODEL_INFO).map(([key, model]) => (
                        <ModelStatusRow
                            key={key}
                            model={model}
                            status={modelStatus[key]}
                        />
                    ))}
                </CardContent>
            </Card>
        </div>
    );
};

export default AnalysisInProgress;
