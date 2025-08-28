// src/components/media/AnalysisInProgress.jsx

import React, { useMemo } from "react";
import { Loader2, Brain, CheckCircle, Clock, AlertTriangle } from "lucide-react";
// UPDATED: Using the new generic progress hook
import { useMediaProgress } from "../../hooks/useMediaProgess.js";
import { useServerStatusQuery } from "../../hooks/useMonitoringQuery.js";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../ui/Card.jsx";
import { DotsSpinner } from "../ui/LoadingSpinner.jsx";

const ModelStatusRow = ({ model, status }) => {
    const statusIcons = {
        PENDING: <Clock className="w-5 h-5 text-gray-400" title="Pending" />,
        PROCESSING: <Loader2 className="w-5 h-5 animate-spin text-yellow-500" title="Processing" />,
        COMPLETED: <CheckCircle className="w-5 h-5 text-green-500" title="Completed" />,
        FAILED: <AlertTriangle className="w-5 h-5 text-red-500" title="Failed" />,
    };
    return (
        <Card className="m-1 flex items-center justify-between p-4 bg-light-muted-background dark:bg-dark-secondary rounded-lg">
            <div className="flex items-center gap-4">
                <div className="flex-shrink-0 w-10 h-10 bg-primary-main/10 rounded-full flex items-center justify-center">
                    <Brain className="w-5 h-5 text-primary-main" />
                </div>
                <div>
                    <p className="font-semibold text-left mb-1">{model.name}</p>
                    <p className="text-xs text-light-muted-text dark:text-dark-muted-text">{model.description}</p>
                </div>
            </div>
            {statusIcons[status]}
        </Card>
    );
};

// RENAMED: Props are now 'media' instead of 'video'
export const AnalysisInProgress = ({ media }) => {
    // UPDATED: Using the new generic progress hook
    const { latestProgress, progressEvents } = useMediaProgress(media.id);
    const { data: serverStatus, isLoading: isServerStatusLoading } = useServerStatusQuery();

    const modelsForMediaType = useMemo(() => {
        if (!serverStatus?.modelsInfo) return [];
        // Filter models from the server that are compatible with the current media type
        return serverStatus.modelsInfo.filter(model => {
            if (media.mediaType === 'VIDEO' || media.mediaType === 'IMAGE') return model.isVideo;
            if (media.mediaType === 'AUDIO') return model.isAudio;
            return false;
        });
    }, [serverStatus, media.mediaType]);

    const modelStatus = useMemo(() => {
        const statusMap = {};
        modelsForMediaType.forEach(model => {
            statusMap[model.name] = "PENDING";
        });

        progressEvents.forEach(event => {
            const modelName = event.data?.modelName;
            if (modelName && statusMap[modelName]) {
                if (event.event === "ANALYSIS_STARTED") statusMap[modelName] = "PROCESSING";
                else if (event.event === "ANALYSIS_COMPLETED") {
                    statusMap[modelName] = event.data.success ? "COMPLETED" : "FAILED";
                }
            }
        });
        return statusMap;
    }, [progressEvents, modelsForMediaType]);

    const latestEventMessage = latestProgress?.message || `Your ${media.mediaType.toLowerCase()} has been queued for analysis...`;
    const formattedStatus = media.status.replace("_", " ").toLowerCase();

    return (
        <div className="mx-auto text-center">
            {/* <Loader2 className="w-16 h-16 text-primary-main mx-auto animate-spin mb-6" /> */}
            <div className="flex items-center justify-center gap-4 mb-2">
            <DotsSpinner/>
                
            </div>
            <h1 className="text-2xl font-bold">Analysis in Progress</h1>
            <p className="text-md text-light-muted-text dark:text-dark-muted-text mb-8">
                Your file "{media.filename}" is being analyzed. This page will update automatically.
            </p>
            <Card>
                <CardHeader>
                    <CardTitle>Current Status: <span className="capitalize text-primary-main">{formattedStatus}</span></CardTitle>
                    <CardDescription>{latestEventMessage}</CardDescription>
                </CardHeader>
                <CardContent className="grid grid-cols-2">
                    {isServerStatusLoading ? (
                        <p>Loading available models...</p>
                    ) : (
                        modelsForMediaType.map((model) => (
                            <ModelStatusRow key={model.name} model={model} status={modelStatus[model.name]} />
                        ))
                    )}
                </CardContent>
            </Card>
        </div>
    );
};

export default AnalysisInProgress;