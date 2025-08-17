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
import { Card } from "../ui/Card";
import { MODEL_INFO } from "../../constants/apiEndpoints.js";

const AnalysisInProgress = ({ video }) => {
    const { latestProgress, progressEvents } = useVideoProgress(video.id);

    // Create a map to track the status of each model's analysis
    const modelStatus = React.useMemo(() => {
        const status = {};

        // Initialize all known models to 'PENDING' based on what the backend supports
        Object.keys(MODEL_INFO).forEach((modelName) => {
            status[modelName] = "PENDING";
        });

        progressEvents.forEach((event) => {
            if (event.data?.modelName) {
                // The event might send the label (e.g., "SigLIP LSTM v3"). We need to find the key ("SIGLIP-LSTM-V3").
                const modelKey = Object.keys(MODEL_INFO).find((key) =>
                    event.data.modelName.includes(key)
                );
                if (modelKey) {
                    if (event.event === "ANALYSIS_STARTED") {
                        status[modelKey] = "PROCESSING";
                    } else if (event.event === "ANALYSIS_COMPLETED") {
                        status[modelKey] = event.data.success
                            ? "COMPLETED"
                            : "FAILED";
                    }
                }
            }
        });
        return status;
    }, [progressEvents]);

    const latestEventMessage =
        latestProgress?.message || "Your video has been queued for analysis...";

    return (
        <div className="max-w-4xl mx-auto text-center">
            <h1 className="text-3xl font-bold mb-2">Analysis in Progress</h1>
            <p className="text-light-muted-text dark:text-dark-muted-text mb-8">
                Your video "{video.filename}" is being analyzed by our AI
                models. This may take a few minutes.
            </p>
            <Card className="p-8">
                <Loader2 className="w-12 h-12 text-primary-main mx-auto animate-spin mb-6" />
                <h2 className="text-xl font-semibold capitalize mb-4">
                    Current Status:{" "}
                    <span className="text-primary-main">
                        {video.status.replace("_", " ")}
                    </span>
                </h2>
                <p className="text-light-muted-text dark:text-dark-muted-text mt-2 mb-8">
                    {latestEventMessage}
                </p>

                <div className="text-left space-y-4">
                    <h3 className="font-semibold text-lg">Model Progress:</h3>
                    {Object.entries(MODEL_INFO).map(([key, model]) => {
                        const status = modelStatus[key];
                        return (
                            <div
                                key={key}
                                className="flex items-center justify-between p-4 bg-light-muted-background dark:bg-dark-muted-background rounded-lg"
                            >
                                <div className="flex items-center gap-3">
                                    <Brain className="w-6 h-6 text-primary-main" />
                                    <div>
                                        <p className="font-semibold">
                                            {model.label}
                                        </p>
                                        <p className="text-xs text-light-muted-text dark:text-dark-muted-text">
                                            {model.description}
                                        </p>
                                    </div>
                                </div>
                                <div>
                                    {status === "PROCESSING" && (
                                        <Loader2
                                            className="w-5 h-5 animate-spin text-yellow-500"
                                            title="Processing"
                                        />
                                    )}
                                    {status === "COMPLETED" && (
                                        <CheckCircle
                                            className="w-5 h-5 text-green-500"
                                            title="Completed"
                                        />
                                    )}
                                    {status === "PENDING" && (
                                        <Clock
                                            className="w-5 h-5 text-gray-400"
                                            title="Pending"
                                        />
                                    )}
                                    {status === "FAILED" && (
                                        <AlertTriangle
                                            className="w-5 h-5 text-red-500"
                                            title="Failed"
                                        />
                                    )}
                                </div>
                            </div>
                        );
                    })}
                </div>
            </Card>
        </div>
    );
};

export default AnalysisInProgress;
