// src/components/videos/AnalysisComplete.jsx

import React from "react";
import { Link } from "react-router-dom";
import { Card } from "../ui/Card";
import { Button } from "../ui/Button";
import { MODEL_INFO } from "../../constants/apiEndpoints";
import { Eye, ShieldCheck, ShieldAlert } from "lucide-react";

const AnalysisCard = ({ analysis, videoId }) => {
    const isReal = analysis.prediction === "REAL";
    const confidence = analysis.confidence * 100;
    const modelInfo = MODEL_INFO[analysis.model];
    const detailsLink = `/results/${videoId}/${analysis.id}`;

    return (
        <Card
            className={`border-2 ${
                isReal ? "border-green-500/30" : "border-red-500/30"
            }`}
        >
            <div className="p-6">
                <div className="flex justify-between items-start mb-4">
                    <div>
                        <h4 className="text-lg font-bold">
                            {modelInfo?.label || analysis.model}
                        </h4>
                        <p className="text-sm text-light-muted-text dark:text-dark-muted-text">
                            {modelInfo?.description}
                        </p>
                    </div>
                    {isReal ? (
                        <ShieldCheck className="w-8 h-8 text-green-500" />
                    ) : (
                        <ShieldAlert className="w-8 h-8 text-red-500" />
                    )}
                </div>
                <div className="flex items-baseline justify-between mb-4">
                    <p className="text-sm">Prediction:</p>
                    <p
                        className={`text-xl font-bold ${
                            isReal ? "text-green-600" : "text-red-600"
                        }`}
                    >
                        {analysis.prediction}
                    </p>
                </div>
                <div className="flex items-baseline justify-between mb-6">
                    <p className="text-sm">Confidence:</p>
                    <p className="text-xl font-bold">
                        {confidence.toFixed(2)}%
                    </p>
                </div>
                <Link to={detailsLink}>
                    <Button variant="outline" className="w-full">
                        <Eye className="mr-2 h-4 w-4" /> View Full Report
                    </Button>
                </Link>
            </div>
        </Card>
    );
};

const AnalysisComplete = ({ video }) => {
    return (
        <div>
            <h2 className="text-3xl font-bold mb-6">Analysis Complete</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {video.analyses
                    .filter((a) => a.status === "COMPLETED") // Only show completed analyses
                    .sort(
                        (a, b) => new Date(b.createdAt) - new Date(a.createdAt)
                    ) // Show newest first
                    .map((analysis) => (
                        <AnalysisCard
                            key={analysis.id}
                            analysis={analysis}
                            videoId={video.id}
                        />
                    ))}
            </div>
        </div>
    );
};

export default AnalysisComplete;
