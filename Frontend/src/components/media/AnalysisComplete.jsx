// src/components/media/AnalysisComplete.jsx

import React from "react";
import { Link } from "react-router-dom";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "../ui/Card.jsx";
import { Button } from "../ui/Button.jsx";
import { useServerStatusQuery } from "../../hooks/useMonitoringQuery.js";
import { Eye, ShieldCheck, ShieldAlert, Clock, AlertTriangle, Bot } from "lucide-react";
import { formatProcessingTime } from "../../utils/formatters.js";
import { EmptyState } from "../ui/EmptyState.jsx";
import { Alert, AlertDescription, AlertTitle } from "../ui/Alert.jsx";

const AnalysisResultCard = ({ analysis, mediaId }) => {
    // The analysis object now contains the full resultPayload.
    const result = analysis.resultPayload;
    const isReal = result.prediction === "REAL";
    const confidence = result.confidence * 100;

    // Card for a FAILED analysis
    if (analysis.status === 'FAILED') {
        return (
             <Card>
                <CardHeader>
                    <CardTitle>{analysis.modelName}</CardTitle>
                    <CardDescription>An AI deepfake detection model.</CardDescription>
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

    return (
        <Card className={`transition-all hover:shadow-lg ${isReal ? "border-green-500/30" : "border-red-500/30"}`}>
            <CardHeader className="flex flex-row items-start justify-between">
                <div>
                    <CardTitle>{analysis.modelName}</CardTitle>
                    <CardDescription>{result.media_type} analysis model.</CardDescription>
                </div>
                {isReal ? <ShieldCheck className="w-8 h-8 text-green-500" /> : <ShieldAlert className="w-8 h-8 text-red-500" />}
            </CardHeader>
            <CardContent className="space-y-4">
                <div className="flex items-baseline justify-between p-4 bg-light-muted-background dark:bg-dark-secondary rounded-lg">
                    <p className="text-sm">Prediction</p>
                    <p className={`text-2xl font-bold ${isReal ? "text-green-600" : "text-red-600"}`}>{result.prediction}</p>
                </div>
                <div className="flex items-baseline justify-between p-4 bg-light-muted-background dark:bg-dark-secondary rounded-lg">
                    <p className="text-sm">Confidence</p>
                    <p className="text-2xl font-bold">{confidence.toFixed(1)}%</p>
                </div>
                <div className="flex items-center justify-between text-xs text-light-muted-text dark:text-dark-muted-text pt-2">
                    <div className="flex items-center gap-2"><Clock className="h-3 w-3" /> Processing Time</div>
                    <span>{formatProcessingTime(result.processingTime)}</span>
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

export const AnalysisComplete = ({ run }) => {
    if (!run?.analyses) {
        return <EmptyState icon={Bot} title="No Results" message="No analysis results were found for this run." />;
    }

    const analyses = [...run.analyses].sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
    const successfulCount = analyses.filter(a => a.status === 'COMPLETED').length;
    const failedCount = analyses.length - successfulCount;

    return (
        <div className="space-y-6">
            <div className="text-center">
                <h2 className="text-3xl font-bold">Analysis Run #{run.runNumber} Complete</h2>
                <p className="text-light-muted-text dark:text-dark-muted-text mt-1">
                    Found {successfulCount} successful result(s) and {failedCount} failed attempt(s).
                </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {analyses.map((analysis) => (
                    <AnalysisResultCard key={analysis.id} analysis={analysis} mediaId={run.mediaId} />
                ))}
            </div>
        </div>
    );
};