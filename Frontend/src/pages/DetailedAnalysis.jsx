// src/pages/DetailedAnalysis.jsx

import React from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import {
    ArrowLeft,
    Brain,
    Clock,
    Activity,
    BarChart3,
    Eye,
    CheckCircle,
    AlertCircle,
    Loader2,
    RefreshCw,
    TrendingUp,
    Cpu,
    Monitor,
    Database,
    LineChart,
} from "lucide-react";
import { useVideoQuery } from "../hooks/useVideosQuery.js";
import { useSpecificAnalysisQuery } from "../hooks/useAnalysisQuery.js";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { PageLoader } from "../components/ui/LoadingSpinner";
import { VideoPlayer } from "../components/videos/VideoPlayer.jsx";
import { ANALYSIS_TYPE_INFO, MODEL_INFO } from "../constants/apiEndpoints.js";

// Helper functions
const formatDate = (dateString) =>
    new Date(dateString).toLocaleString("en-US", {
        year: "numeric",
        month: "long",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
    });

const formatProcessingTime = (timeInSeconds) => {
    if (!timeInSeconds) return "N/A";
    if (timeInSeconds < 60) return `${timeInSeconds.toFixed(1)}s`;
    const minutes = Math.floor(timeInSeconds / 60);
    const seconds = (timeInSeconds % 60).toFixed(1);
    return `${minutes}m ${seconds}s`;
};

const getConfidenceColor = (confidence) => {
    const percentage = confidence * 100;
    if (percentage >= 80) return "text-green-600";
    if (percentage >= 60) return "text-yellow-600";
    return "text-red-600";
};

// Main Result Display Component
const AnalysisResultCard = ({ analysis }) => {
    if (!analysis) return null;

    const isReal = analysis.prediction === "REAL";
    const confidence = analysis.confidence * 100;

    return (
        <Card
            className={`border-2 ${
                isReal ? "border-green-500/30" : "border-red-500/30"
            }`}
        >
            <div className="text-center mb-8">
                <div
                    className={`inline-flex items-center justify-center w-16 h-16 rounded-full mb-4 ${
                        isReal
                            ? "bg-green-100 dark:bg-green-900/30"
                            : "bg-red-100 dark:bg-red-900/30"
                    }`}
                >
                    {isReal ? (
                        <CheckCircle className="h-8 w-8 text-green-600" />
                    ) : (
                        <AlertCircle className="h-8 w-8 text-red-600" />
                    )}
                </div>

                <h2
                    className={`text-4xl font-bold mb-2 ${
                        isReal ? "text-green-600" : "text-red-600"
                    }`}
                >
                    {confidence.toFixed(1)}%
                </h2>

                <p className="text-xl font-semibold mb-1">
                    {isReal ? "Authentic Content" : "Potential Deepfake"}
                </p>

                <p className="text-sm text-gray-600 dark:text-gray-400">
                    Confidence Score
                </p>
            </div>

            {/* Processing Information */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-6 border-t border-gray-200 dark:border-gray-700">
                <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                        Processing Time:
                    </span>
                    <span className="text-sm font-medium">
                        {formatProcessingTime(analysis.processingTime)}
                    </span>
                </div>
                <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                        Analyzed:
                    </span>
                    <span className="text-sm font-medium">
                        {formatDate(analysis.createdAt)}
                    </span>
                </div>
            </div>
        </Card>
    );
};

// Enhanced Analysis Details Component
const AnalysisDetailsCard = ({ details }) => {
    if (!details) return null;

    return (
        <Card>
            <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                <Database className="h-5 w-5" />
                Analysis Details
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-3">
                    <div className="flex justify-between">
                        <span className="text-sm text-gray-600">
                            Frame Count:
                        </span>
                        <span className="text-sm font-medium">
                            {details.frameCount}
                        </span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-sm text-gray-600">
                            Average Confidence:
                        </span>
                        <span className="text-sm font-medium">
                            {(details.avgConfidence * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-sm text-gray-600">
                            Confidence Std Dev:
                        </span>
                        <span className="text-sm font-medium">
                            {(details.confidenceStd * 100).toFixed(1)}%
                        </span>
                    </div>
                </div>
                <div className="space-y-3">
                    {details.temporalConsistency && (
                        <div className="flex justify-between">
                            <span className="text-sm text-gray-600">
                                Temporal Consistency:
                            </span>
                            <span className="text-sm font-medium">
                                {(details.temporalConsistency * 100).toFixed(1)}
                                %
                            </span>
                        </div>
                    )}
                    {details.rollingAverage && (
                        <div className="flex justify-between">
                            <span className="text-sm text-gray-600">
                                Rolling Average:
                            </span>
                            <span className="text-sm font-medium">
                                {(details.rollingAverage * 100).toFixed(1)}%
                            </span>
                        </div>
                    )}
                </div>
            </div>
        </Card>
    );
};

// Frame Analysis Component
const FrameAnalysisCard = ({ frames }) => {
    if (!frames || frames.length === 0) return null;

    return (
        <Card>
            <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                <LineChart className="h-5 w-5" />
                Frame-by-Frame Analysis ({frames.length} frames)
            </h3>
            <div className="space-y-4">
                {/* Summary Statistics */}
                <div className="grid grid-cols-3 gap-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                    <div className="text-center">
                        <div className="text-2xl font-bold text-green-600">
                            {
                                frames.filter((f) => f.prediction === "REAL")
                                    .length
                            }
                        </div>
                        <div className="text-sm text-gray-600">Real Frames</div>
                    </div>
                    <div className="text-center">
                        <div className="text-2xl font-bold text-red-600">
                            {
                                frames.filter((f) => f.prediction === "FAKE")
                                    .length
                            }
                        </div>
                        <div className="text-sm text-gray-600">Fake Frames</div>
                    </div>
                    <div className="text-center">
                        <div className="text-2xl font-bold text-blue-600">
                            {(
                                (frames.reduce(
                                    (sum, f) => sum + f.confidence,
                                    0
                                ) /
                                    frames.length) *
                                100
                            ).toFixed(1)}
                            %
                        </div>
                        <div className="text-sm text-gray-600">
                            Avg Confidence
                        </div>
                    </div>
                </div>

                {/* Sample Frame Data */}
                <div className="max-h-64 overflow-y-auto">
                    <table className="w-full text-sm">
                        <thead className="bg-gray-50 dark:bg-gray-800">
                            <tr>
                                <th className="text-left p-2">Frame #</th>
                                <th className="text-left p-2">Prediction</th>
                                <th className="text-left p-2">Confidence</th>
                                <th className="text-left p-2">Timestamp</th>
                            </tr>
                        </thead>
                        <tbody>
                            {frames.slice(0, 20).map((frame, index) => (
                                <tr key={index} className="border-t">
                                    <td className="p-2">{frame.frameNumber}</td>
                                    <td className="p-2">
                                        <span
                                            className={`px-2 py-1 rounded text-xs ${
                                                frame.prediction === "REAL"
                                                    ? "bg-green-100 text-green-800"
                                                    : "bg-red-100 text-red-800"
                                            }`}
                                        >
                                            {frame.prediction}
                                        </span>
                                    </td>
                                    <td className="p-2">
                                        {(frame.confidence * 100).toFixed(1)}%
                                    </td>
                                    <td className="p-2">
                                        {frame.timestamp
                                            ? `${frame.timestamp.toFixed(2)}s`
                                            : "N/A"}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                    {frames.length > 20 && (
                        <p className="text-sm text-gray-500 text-center mt-2">
                            Showing first 20 of {frames.length} frames
                        </p>
                    )}
                </div>
            </div>
        </Card>
    );
};

// Temporal Analysis Component
const TemporalAnalysisCard = ({ temporal }) => {
    if (!temporal) return null;

    return (
        <Card>
            <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                Temporal Analysis
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                    <div className="flex justify-between">
                        <span className="text-sm text-gray-600">
                            Consistency Score:
                        </span>
                        <span className="text-sm font-medium">
                            {(temporal.consistencyScore * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-sm text-gray-600">
                            Total Frames:
                        </span>
                        <span className="text-sm font-medium">
                            {temporal.totalFrames}
                        </span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-sm text-gray-600">
                            Real Frames:
                        </span>
                        <span className="text-sm font-medium text-green-600">
                            {temporal.realFrames}
                        </span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-sm text-gray-600">
                            Fake Frames:
                        </span>
                        <span className="text-sm font-medium text-red-600">
                            {temporal.fakeFrames}
                        </span>
                    </div>
                </div>
                <div className="space-y-4">
                    <div className="flex justify-between">
                        <span className="text-sm text-gray-600">
                            Average Confidence:
                        </span>
                        <span className="text-sm font-medium">
                            {(temporal.avgConfidence * 100).toFixed(1)}%
                        </span>
                    </div>
                    {temporal.patternDetection && (
                        <div>
                            <span className="text-sm text-gray-600">
                                Pattern Detection:
                            </span>
                            <p className="text-sm mt-1">
                                {temporal.patternDetection}
                            </p>
                        </div>
                    )}
                    {temporal.confidenceTrend && (
                        <div>
                            <span className="text-sm text-gray-600">
                                Confidence Trend:
                            </span>
                            <p className="text-sm mt-1">
                                {temporal.confidenceTrend}
                            </p>
                        </div>
                    )}
                    {temporal.anomalyFrames &&
                        temporal.anomalyFrames.length > 0 && (
                            <div>
                                <span className="text-sm text-gray-600">
                                    Anomaly Frames:
                                </span>
                                <p className="text-sm mt-1">
                                    {temporal.anomalyFrames.join(", ")}
                                </p>
                            </div>
                        )}
                </div>
            </div>
        </Card>
    );
};

// Model Info Component
const ModelInfoCard = ({ modelInfo }) => {
    if (!modelInfo) return null;

    return (
        <Card>
            <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                <Cpu className="h-5 w-5" />
                Model Information
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-3">
                    <div className="flex justify-between">
                        <span className="text-sm text-gray-600">Version:</span>
                        <span className="text-sm font-medium">
                            {modelInfo.version}
                        </span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-sm text-gray-600">
                            Architecture:
                        </span>
                        <span className="text-sm font-medium">
                            {modelInfo.architecture}
                        </span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-sm text-gray-600">Device:</span>
                        <span className="text-sm font-medium">
                            {modelInfo.device}
                        </span>
                    </div>
                </div>
                <div className="space-y-3">
                    {modelInfo.batchSize && (
                        <div className="flex justify-between">
                            <span className="text-sm text-gray-600">
                                Batch Size:
                            </span>
                            <span className="text-sm font-medium">
                                {modelInfo.batchSize}
                            </span>
                        </div>
                    )}
                    {modelInfo.numFrames && (
                        <div className="flex justify-between">
                            <span className="text-sm text-gray-600">
                                Frames Processed:
                            </span>
                            <span className="text-sm font-medium">
                                {modelInfo.numFrames}
                            </span>
                        </div>
                    )}
                </div>
            </div>
        </Card>
    );
};

// System Info Component
const SystemInfoCard = ({ systemInfo }) => {
    if (!systemInfo) return null;

    return (
        <Card>
            <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                <Monitor className="h-5 w-5" />
                System Information
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-3">
                    {systemInfo.processingDevice && (
                        <div className="flex justify-between">
                            <span className="text-sm text-gray-600">
                                Processing Device:
                            </span>
                            <span className="text-sm font-medium">
                                {systemInfo.processingDevice}
                            </span>
                        </div>
                    )}
                    {systemInfo.cudaAvailable !== undefined && (
                        <div className="flex justify-between">
                            <span className="text-sm text-gray-600">
                                CUDA Available:
                            </span>
                            <span
                                className={`text-sm font-medium ${
                                    systemInfo.cudaAvailable
                                        ? "text-green-600"
                                        : "text-red-600"
                                }`}
                            >
                                {systemInfo.cudaAvailable ? "Yes" : "No"}
                            </span>
                        </div>
                    )}
                    {systemInfo.gpuMemoryUsed && (
                        <div className="flex justify-between">
                            <span className="text-sm text-gray-600">
                                GPU Memory Used:
                            </span>
                            <span className="text-sm font-medium">
                                {systemInfo.gpuMemoryUsed}
                            </span>
                        </div>
                    )}
                </div>
                <div className="space-y-3">
                    {systemInfo.systemMemoryUsed && (
                        <div className="flex justify-between">
                            <span className="text-sm text-gray-600">
                                System Memory Used:
                            </span>
                            <span className="text-sm font-medium">
                                {systemInfo.systemMemoryUsed}
                            </span>
                        </div>
                    )}
                    {systemInfo.loadBalancingInfo && (
                        <div>
                            <span className="text-sm text-gray-600">
                                Load Balancing:
                            </span>
                            <pre className="text-xs mt-1 p-2 bg-gray-100 dark:bg-gray-800 rounded">
                                {JSON.stringify(
                                    systemInfo.loadBalancingInfo,
                                    null,
                                    2
                                )}
                            </pre>
                        </div>
                    )}
                </div>
            </div>
        </Card>
    );
};

// Main Detailed Analysis Component
const DetailedAnalysis = () => {
    const { videoId, modelId } = useParams();
    const navigate = useNavigate();

    // Parse modelId to extract type and model (format: "QUICK-SIGLIP_LSTM_V1")
    const [analysisType, model] = modelId?.split("-") || [];

    const {
        data: video,
        isLoading: isVideoLoading,
        error: videoError,
        refetch: refetchVideo,
    } = useVideoQuery(videoId);

    const {
        data: analysis,
        isLoading: isAnalysisLoading,
        error: analysisError,
        refetch: refetchAnalysis,
    } = useSpecificAnalysisQuery(videoId, analysisType, model);

    const handleRefresh = () => {
        refetchVideo();
        refetchAnalysis();
    };

    // Loading state
    if (isVideoLoading || isAnalysisLoading) {
        return <PageLoader text="Loading detailed analysis..." />;
    }

    // Error state
    if (videoError || analysisError) {
        return (
            <div className="text-center p-8 max-w-md mx-auto">
                <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
                <h2 className="text-2xl font-bold mb-2">
                    Error Loading Analysis
                </h2>
                <p className="text-gray-600 dark:text-gray-400 mb-6">
                    {videoError?.message ||
                        analysisError?.message ||
                        "Failed to load analysis data"}
                </p>
                <div className="space-x-2">
                    <Button onClick={handleRefresh} variant="outline">
                        <RefreshCw className="mr-2 h-4 w-4" />
                        Try Again
                    </Button>
                    <Button onClick={() => navigate(`/results/${videoId}`)}>
                        <ArrowLeft className="mr-2 h-4 w-4" />
                        Back to Results
                    </Button>
                </div>
            </div>
        );
    }

    // No analysis found
    if (!analysis) {
        return (
            <div className="text-center p-8 max-w-md mx-auto">
                <Eye className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <h2 className="text-2xl font-bold mb-2">Analysis Not Found</h2>
                <p className="text-gray-600 dark:text-gray-400 mb-6">
                    The requested analysis ({analysisType} with {model}) was not
                    found.
                </p>
                <Button onClick={() => navigate(`/results/${videoId}`)}>
                    <ArrowLeft className="mr-2 h-4 w-4" />
                    Back to Results
                </Button>
            </div>
        );
    }

    const typeInfo = ANALYSIS_TYPE_INFO[analysisType];
    const modelInfo = MODEL_INFO[model];

    return (
        <div className="space-y-6 mx-auto">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <Link to={`/results/${videoId}`}>
                        <Button variant="outline" size="sm">
                            <ArrowLeft className="h-4 w-4 mr-2" />
                            Back to Results
                        </Button>
                    </Link>
                    <div>
                        <h1 className="text-2xl font-bold flex items-center gap-2">
                            <span className="text-lg">{typeInfo?.icon}</span>
                            {typeInfo?.label || analysisType} Analysis
                        </h1>
                        <p className="text-gray-600 dark:text-gray-400">
                            Model: {modelInfo?.label || model}
                        </p>
                    </div>
                </div>
                <Button onClick={handleRefresh} variant="outline" size="sm">
                    <RefreshCw className="mr-2 h-4 w-4" />
                    Refresh
                </Button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Video and Basic Info */}
                <div className="lg:col-span-1 space-y-6">
                    {/* Video Player */}
                    <Card>
                        <div className="aspect-video rounded-lg overflow-hidden bg-black">
                            <VideoPlayer videoUrl={video.url} />
                        </div>
                        <div className="mt-4">
                            <h3 className="font-semibold">{video.filename}</h3>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                                {video.description}
                            </p>
                        </div>
                    </Card>

                    {/* Analysis Result */}
                    <AnalysisResultCard analysis={analysis} />
                </div>

                {/* Detailed Analysis Data */}
                <div className="lg:col-span-2 space-y-6">
                    {/* Enhanced Analysis Data */}
                    {analysis.analysisDetails && (
                        <AnalysisDetailsCard
                            details={analysis.analysisDetails}
                        />
                    )}

                    {/* Frame Analysis Data */}
                    {analysis.frameAnalysis &&
                        analysis.frameAnalysis.length > 0 && (
                            <FrameAnalysisCard
                                frames={analysis.frameAnalysis}
                            />
                        )}

                    {/* Temporal Analysis Data */}
                    {analysis.temporalAnalysis && (
                        <TemporalAnalysisCard
                            temporal={analysis.temporalAnalysis}
                        />
                    )}

                    {/* Model Information */}
                    {analysis.modelInfo && (
                        <ModelInfoCard modelInfo={analysis.modelInfo} />
                    )}

                    {/* System Information */}
                    {analysis.systemInfo && (
                        <SystemInfoCard systemInfo={analysis.systemInfo} />
                    )}

                    {/* Visualization (if available) */}
                    {analysis.visualizedUrl && (
                        <Card>
                            <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                                <TrendingUp className="h-5 w-5" />
                                Analysis Visualization
                            </h3>
                            <div className="aspect-video rounded-lg overflow-hidden bg-black">
                                <VideoPlayer
                                    videoUrl={analysis.visualizedUrl}
                                />
                            </div>
                        </Card>
                    )}

                    {/* Error Information (if any) */}
                    {analysis.errorMessage && (
                        <Card className="border-red-200 dark:border-red-800">
                            <h3 className="text-lg font-bold mb-4 flex items-center gap-2 text-red-600">
                                <AlertCircle className="h-5 w-5" />
                                Analysis Error
                            </h3>
                            <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
                                <p className="text-sm text-red-700 dark:text-red-300">
                                    {analysis.errorMessage}
                                </p>
                            </div>
                        </Card>
                    )}
                </div>
            </div>
        </div>
    );
};

export default DetailedAnalysis;
