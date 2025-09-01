// src/components/media/AnalysisInProgress.jsx

import React, { useMemo } from "react";
import { Loader2, Brain, CheckCircle, Clock, AlertTriangle } from "lucide-react";
import { useMediaProgress } from "../../hooks/useMediaProgess.js";
import { useServerStatusQuery } from "../../hooks/useMonitoringQuery.js";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../ui/Card.jsx";
import { DotsSpinner } from "../ui/LoadingSpinner.jsx";
import { getMediaType } from "../../utils/media.js";

const ModelStatusRow = ({ model, status }) => { /* ... (component unchanged) ... */ };

export const AnalysisInProgress = ({ media }) => {
    const { latestProgress, progressEvents } = useMediaProgress(media.id);
    const { data: serverStatus, isLoading: isServerStatusLoading } = useServerStatusQuery();

    const modelsForThisRun = useMemo(() => {
        if (!serverStatus?.models_info) return [];
        const mediaType = getMediaType(media.mimetype);
        return serverStatus.models_info.filter(model => {
            if (!model.loaded) return false;
            if (mediaType === 'VIDEO' || mediaType === 'IMAGE') return model.isVideo;
            if (mediaType === 'AUDIO') return model.isAudio;
            return false;
        });
    }, [serverStatus, media.mimetype]);
    
    const modelStatus = useMemo(() => {
        const statusMap = {};
        modelsForThisRun.forEach(model => { statusMap[model.name] = "PENDING"; });

        // The Python server emits events with `model_name`.
        progressEvents.forEach(event => {
            const modelName = event.data?.model_name;
            if (modelName && statusMap[modelName]) {
                if (event.event.includes("START")) statusMap[modelName] = "PROCESSING";
                if (event.event.includes("COMPLETE")) statusMap[modelName] = "COMPLETED";
            }
        });
        return statusMap;
    }, [progressEvents, modelsForThisRun]);

    const latestEventMessage = latestProgress?.message || `Your ${media.mediaType.toLowerCase()} is being queued for analysis...`;

    return (
        <div className="mx-auto text-center max-w-4xl">
            <DotsSpinner size="lg" />
            <h1 className="text-3xl font-bold mt-4">Analysis in Progress</h1>
            <p className="text-lg text-light-muted-text dark:text-dark-muted-text mt-2 mb-8">
                Your file "{media.filename}" is being analyzed. This page will update automatically.
            </p>
            <Card>
                <CardHeader>
                    <CardTitle>Current Status: <span className="capitalize text-primary-main">{media.status.replace("_", " ").toLowerCase()}</span></CardTitle>
                    <CardDescription>{latestEventMessage}</CardDescription>
                </CardHeader>
                <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    {isServerStatusLoading ? <p>Loading model status...</p> : (
                        modelsForThisRun.map((model) => (
                            <ModelStatusRow key={model.name} model={model} status={modelStatus[model.name]} />
                        ))
                    )}
                </CardContent>
            </Card>
        </div>
    );
};