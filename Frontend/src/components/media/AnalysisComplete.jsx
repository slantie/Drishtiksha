// src/components/media/AnalysisComplete.jsx

import React from "react";
import { Link } from "react-router-dom";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "../ui/Card.jsx";
import { Button } from "../ui/Button.jsx";
import { useServerStatusQuery } from "../../hooks/useMonitoringQuery.js";
import { Eye, ShieldCheck, ShieldAlert, Clock, AlertTriangle } from "lucide-react";
import { formatProcessingTime } from "../../utils/formatters.js";
import { EmptyState } from "../ui/EmptyState.jsx";
import { Bot } from "lucide-react";

// This card is already quite generic, but we will ensure props are clear.
const AnalysisResultCard = ({ analysis, mediaId, serverModels }) => {
    const isReal = analysis.prediction === "REAL";
    const confidence = analysis.confidence * 100;

    const modelInfo = serverModels?.find((m) => m.name === analysis.model) || {
        name: analysis.model,
        description: "A deepfake detection model.",
    };

    // Card for a FAILED analysis
    if (analysis.status === 'FAILED') {
        return (
             <Card className="border-gray-500/30">
                <CardHeader>
                    <CardTitle>{modelInfo.name}</CardTitle>
                    <CardDescription>{modelInfo.description}</CardDescription>
                </CardHeader>
                <CardContent>
                    <Alert variant="destructive">
                        <AlertTriangle className="h-4 w-4" />
                        <AlertTitle>Analysis Failed</AlertTitle>
                        <AlertDescription>{analysis.errorMessage || "An unknown error occurred."}</AlertDescription>
                    </Alert>
                </CardContent>
            </Card>
        )
    }

    // Card for a COMPLETED analysis
    return (
        <Card className={`transition-all hover:shadow-lg ${isReal ? "border-green-500/30" : "border-red-500/30"}`}>
            <CardHeader className="flex flex-row items-start justify-between">
                <div>
                    <CardTitle>{modelInfo.name}</CardTitle>
                    <CardDescription>{modelInfo.description}</CardDescription>
                </div>
                {isReal ? <ShieldCheck className="w-8 h-8 text-green-500 flex-shrink-0" /> : <ShieldAlert className="w-8 h-8 text-red-500 flex-shrink-0" />}
            </CardHeader>
            <CardContent className="space-y-4">
                <div className="flex items-baseline justify-between p-4 bg-light-muted-background dark:bg-dark-secondary rounded-lg">
                    <p className="text-sm text-light-muted-text dark:text-dark-muted-text">Prediction</p>
                    <p className={`text-2xl font-bold ${isReal ? "text-green-600" : "text-red-600"}`}>{analysis.prediction}</p>
                </div>
                <div className="flex items-baseline justify-between p-4 bg-light-muted-background dark:bg-dark-secondary rounded-lg">
                    <p className="text-sm text-light-muted-text dark:text-dark-muted-text">Confidence</p>
                    <p className="text-2xl font-bold">{confidence.toFixed(1)}%</p>
                </div>
                <div className="flex items-center justify-between text-xs text-light-muted-text dark:text-dark-muted-text pt-2">
                    <div className="flex items-center gap-2"><Clock className="h-3 w-3" /> Processing Time</div>
                    <span className="font-sans">{formatProcessingTime(analysis.processingTime)}</span>
                </div>
            </CardContent>
            <CardFooter>
                 <Button asChild variant="outline" className="w-full">
                    <Link to={`/results/${mediaId}/${analysis.id}`}><Eye className="mr-2 h-4 w-4" /> View Full Report</Link>
                </Button>
            </CardFooter>
        </Card>
    );
};

// RENAMED: Prop is now 'media'
export const AnalysisComplete = ({ media }) => {
    const { data: serverStatus } = useServerStatusQuery();
    const serverModels = serverStatus?.modelsInfo || [];

    const completedAnalyses = media.analyses.filter(a => a.status === "COMPLETED").sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
    const failedAnalyses = media.analyses.filter(a => a.status === "FAILED");

    if (completedAnalyses.length === 0 && failedAnalyses.length === 0) {
        return (
            <EmptyState
                icon={Bot}
                title="Analysis Pending"
                message="The analysis process has completed, but no results were returned. This may indicate an issue with the processing queue."
            />
        )
    }

    return (
        <div className="space-y-6">
            <div className="text-center">
                <h2 className="text-3xl font-bold">Analysis Complete</h2>
                <p className="text-light-muted-text dark:text-dark-muted-text mt-1">
                    Found {completedAnalyses.length} successful result(s) and {failedAnalyses.length} failed attempt(s).
                </p>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {completedAnalyses.map((analysis) => (
                    <AnalysisResultCard key={analysis.id} analysis={analysis} mediaId={media.id} serverModels={serverModels} />
                ))}
                 {failedAnalyses.map((analysis) => (
                    <AnalysisResultCard key={analysis.id} analysis={analysis} mediaId={media.id} serverModels={serverModels} />
                ))}
            </div>
        </div>
    );
};

export default AnalysisComplete;