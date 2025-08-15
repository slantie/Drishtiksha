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
    Layers,
    CheckCircle,
    AlertCircle,
    Loader2,
    RefreshCw,
    Download,
    TrendingUp,
    Cpu,
    HardDrive,
    Zap,
    Monitor,
    Database,
    LineChart,
    PieChart,
    BarChart,
    Settings,
} from "lucide-react";
import { useVideoQuery } from "../hooks/useVideosQuery.js";
import {
    useSpecificAnalysisQuery,
} from "../hooks/useAnalysisQuery.js";
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
    if (percentage >= 90) return "text-green-600";
    if (percentage >= 70) return "text-yellow-600";
    return "text-red-600";
};

const getStatusColor = (status) => {
    const statusColors = {
        PENDING: "bg-gray-500/10 text-gray-600 border-gray-200",
        PROCESSING: "bg-yellow-500/10 text-yellow-600 border-yellow-200",
        COMPLETED: "bg-green-500/10 text-green-600 border-green-200",
        FAILED: "bg-red-500/10 text-red-600 border-red-200",
    };
    return (
        statusColors[status] || "bg-gray-500/10 text-gray-600 border-gray-200"
    );
};

// Analysis Header Component
const AnalysisHeader = ({
    video,
    analysis,
    analysisType,
    model,
    onRefresh,
}) => {
    const typeInfo = ANALYSIS_TYPE_INFO[analysisType];
    const modelInfo = MODEL_INFO[model];

    return (
        <Card>
            <div className="flex flex-col space-y-4">
                {/* Navigation and Title */}
                <div className="flex items-center gap-3">
                    <Link
                        to={`/results/${video.id}`}
                        className="text-blue-600 hover:text-blue-800 bg-blue-300/30 dark:bg-blue-800/30 rounded-full p-2"
                    >
                        <ArrowLeft className="h-6 w-6" />
                    </Link>
                    <div className="flex-1">
                        <h1 className="text-3xl font-bold flex items-center gap-3">
                            <span>{typeInfo?.icon}</span>
                            {typeInfo?.label} Results
                        </h1>
                        <p className="text-light-muted-text dark:text-dark-muted-text mt-1">
                            {video.filename} • {modelInfo?.label}
                        </p>
                    </div>
                    <Button onClick={onRefresh} variant="outline">
                        <RefreshCw className="mr-2 h-5 w-5" />
                        Refresh
                    </Button>
                </div>

                {/* Analysis Status */}
                <div className="flex flex-wrap items-center gap-4 pt-2 border-t border-gray-200 dark:border-gray-700">
                    <div className="flex items-center gap-2">
                        <Activity className="h-4 w-4 text-gray-500" />
                        <span className="text-sm text-gray-600 dark:text-gray-400">
                            Status:
                        </span>
                        <span
                            className={`px-2 py-1 rounded-full text-xs font-semibold border ${getStatusColor(
                                analysis?.status
                            )}`}
                        >
                            {analysis?.status || "UNKNOWN"}
                        </span>
                    </div>
                    <div className="flex items-center gap-2">
                        <Brain className="h-4 w-4 text-gray-500" />
                        <span className="text-sm text-gray-600 dark:text-gray-400">
                            Model:
                        </span>
                        <span className="text-sm font-medium">
                            {modelInfo?.label}
                        </span>
                    </div>
                    <div className="flex items-center gap-2">
                        <Clock className="h-4 w-4 text-gray-500" />
                        <span className="text-sm text-gray-600 dark:text-gray-400">
                            Duration:
                        </span>
                        <span className="text-sm font-medium">
                            {typeInfo?.duration}
                        </span>
                    </div>
                </div>
            </div>
        </Card>
    );
};

// Main Result Display Component
const AnalysisResultCard = ({ analysis, analysisType }) => {
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

// Detailed Metrics Component (for DETAILED analysis)
const DetailedMetricsCard = ({ analysis }) => {
    if (!analysis?.analysisDetails) return null;

    const metrics = analysis.analysisDetails;

    return (
        <Card>
            <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Detailed Metrics
            </h3>

            <div className="space-y-4">
                {metrics.frameConsistency !== undefined && (
                    <div className="space-y-2">
                        <div className="flex justify-between items-center">
                            <span className="text-sm font-medium">
                                Frame Consistency
                            </span>
                            <span
                                className={`text-sm font-bold ${getConfidenceColor(
                                    metrics.frameConsistency
                                )}`}
                            >
                                {(metrics.frameConsistency * 100).toFixed(1)}%
                            </span>
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                            <div
                                className="bg-blue-600 h-2 rounded-full transition-all"
                                style={{
                                    width: `${metrics.frameConsistency * 100}%`,
                                }}
                            />
                        </div>
                    </div>
                )}

                {metrics.temporalCoherence !== undefined && (
                    <div className="space-y-2">
                        <div className="flex justify-between items-center">
                            <span className="text-sm font-medium">
                                Temporal Coherence
                            </span>
                            <span
                                className={`text-sm font-bold ${getConfidenceColor(
                                    metrics.temporalCoherence
                                )}`}
                            >
                                {(metrics.temporalCoherence * 100).toFixed(1)}%
                            </span>
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                            <div
                                className="bg-green-600 h-2 rounded-full transition-all"
                                style={{
                                    width: `${
                                        metrics.temporalCoherence * 100
                                    }%`,
                                }}
                            />
                        </div>
                    </div>
                )}

                {metrics.facialArtifacts !== undefined && (
                    <div className="space-y-2">
                        <div className="flex justify-between items-center">
                            <span className="text-sm font-medium">
                                Facial Artifacts
                            </span>
                            <span
                                className={`text-sm font-bold ${getConfidenceColor(
                                    1 - metrics.facialArtifacts
                                )}`}
                            >
                                {(metrics.facialArtifacts * 100).toFixed(1)}%
                            </span>
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                            <div
                                className="bg-red-600 h-2 rounded-full transition-all"
                                style={{
                                    width: `${metrics.facialArtifacts * 100}%`,
                                }}
                            />
                        </div>
                    </div>
                )}
            </div>
        </Card>
    );
};

// Frame Analysis Component (for FRAMES analysis)
const FrameAnalysisCard = ({ analysis }) => {
    if (!analysis?.frameAnalyses?.length) return null;

    const frameAnalyses = analysis.frameAnalyses.slice(0, 10); // Show first 10 frames

    return (
        <Card>
            <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                <Layers className="h-5 w-5" />
                Frame-by-Frame Analysis
            </h3>

            <div className="space-y-3">
                {frameAnalyses.map((frame, index) => (
                    <div
                        key={index}
                        className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg"
                    >
                        <div className="flex items-center gap-3">
                            <span className="text-sm font-medium">
                                Frame {frame.frameNumber}
                            </span>
                            <span
                                className={`px-2 py-1 rounded text-xs font-semibold ${
                                    frame.prediction === "REAL"
                                        ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300"
                                        : "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300"
                                }`}
                            >
                                {frame.prediction}
                            </span>
                        </div>
                        <span
                            className={`text-sm font-bold ${getConfidenceColor(
                                frame.confidence
                            )}`}
                        >
                            {(frame.confidence * 100).toFixed(1)}%
                        </span>
                    </div>
                ))}
            </div>

            {analysis.frameAnalyses.length > 10 && (
                <div className="mt-4 text-center">
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                        Showing 10 of {analysis.frameAnalyses.length} frames
                        analyzed
                    </p>
                </div>
            )}
        </Card>
    );
};

// Model Information Component
const ModelInfoCard = ({ analysis }) => {
    if (!analysis?.modelInfo) return null;

    const modelInfo = analysis.modelInfo;

    return (
        <Card>
            <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                <Cpu className="h-5 w-5" />
                Model Information
            </h3>

            <div className="space-y-3">
                <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                        Model Name:
                    </span>
                    <span className="text-sm font-medium">
                        {modelInfo.name}
                    </span>
                </div>

                {modelInfo.version && (
                    <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600 dark:text-gray-400">
                            Version:
                        </span>
                        <span className="text-sm font-medium">
                            {modelInfo.version}
                        </span>
                    </div>
                )}

                {modelInfo.architecture && (
                    <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600 dark:text-gray-400">
                            Architecture:
                        </span>
                        <span className="text-sm font-medium">
                            {modelInfo.architecture}
                        </span>
                    </div>
                )}
            </div>
        </Card>
    );
};

// System Information Component
const SystemInfoCard = ({ analysis }) => {
    if (!analysis?.systemInfo) return null;

    const systemInfo = analysis.systemInfo;

    return (
        <Card>
            <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                <Activity className="h-5 w-5" />
                System Information
            </h3>

            <div className="space-y-3">
                {systemInfo.gpu && (
                    <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600 dark:text-gray-400">
                            GPU:
                        </span>
                        <span className="text-sm font-medium">
                            {systemInfo.gpu}
                        </span>
                    </div>
                )}

                {systemInfo.memory && (
                    <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600 dark:text-gray-400">
                            Memory:
                        </span>
                        <span className="text-sm font-medium">
                            {systemInfo.memory} MB
                        </span>
                    </div>
                )}

                {systemInfo.processingTime && (
                    <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600 dark:text-gray-400">
                            Processing Time:
                        </span>
                        <span className="text-sm font-medium">
                            {formatProcessingTime(systemInfo.processingTime)}
                        </span>
                    </div>
                )}
            </div>
        </Card>
    );
};

// Main Detailed Analysis Component
export const DetailedAnalysis = () => {
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
                <p className="text-light-muted-text dark:text-dark-muted-text mb-6">
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
                <p className="text-light-muted-text dark:text-dark-muted-text mb-6">
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

    return (
        <div className="space-y-6 mx-auto">
            {/* Header */}
            <AnalysisHeader
                video={video}
                analysis={analysis}
                analysisType={analysisType}
                model={model}
                onRefresh={handleRefresh}
            />

            {/* Main Content Grid */}
            <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
                {/* Left Column - Video and Main Result */}
                <div className="xl:col-span-1 space-y-6">
                    {/* Video Player */}
                    <Card>
                        <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                            <Eye className="h-5 w-5" />
                            Original Video
                        </h3>
                        <div className="aspect-video rounded-lg overflow-hidden bg-black">
                            <VideoPlayer videoUrl={video.url} />
                        </div>
                    </Card>

                    {/* Main Analysis Result */}
                    <AnalysisResultCard
                        analysis={analysis}
                        analysisType={analysisType}
                    />
                </div>

                {/* Right Column - Detailed Information */}
                <div className="xl:col-span-2 space-y-6">
                    {/* Detailed Metrics (for DETAILED analysis) */}
                    {analysisType === "DETAILED" && (
                        <DetailedMetricsCard analysis={analysis} />
                    )}

                    {/* Frame Analysis (for FRAMES analysis) */}
                    {analysisType === "FRAMES" && (
                        <FrameAnalysisCard analysis={analysis} />
                    )}

                    {/* Model Information */}
                    <ModelInfoCard analysis={analysis} />

                    {/* System Information */}
                    <SystemInfoCard analysis={analysis} />

                    {/* Enhanced Analysis Data */}
                    {analysis.analysisDetails && (
                        <AnalysisDetailsCard details={analysis.analysisDetails} />
                    )}

                    {/* Frame Analysis Data */}
                    {analysis.frameAnalysis && analysis.frameAnalysis.length > 0 && (
                        <FrameAnalysisCard frames={analysis.frameAnalysis} />
                    )}

                    {/* Temporal Analysis Data */}
                    {analysis.temporalAnalysis && (
                        <TemporalAnalysisCard temporal={analysis.temporalAnalysis} />
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
                    {analysis.analysisError && (
                        <Card className="border-red-200 dark:border-red-800">
                            <h3 className="text-lg font-bold mb-4 flex items-center gap-2 text-red-600">
                                <AlertCircle className="h-5 w-5" />
                                Analysis Error
                            </h3>
                            <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
                                <p className="text-sm text-red-700 dark:text-red-300">
                                    {analysis.analysisError}
                                </p>
                            </div>
                        </Card>
                    )}
                </div>
            </div>
        </div>
    );
};

// Enhanced Analysis Details Component
const AnalysisDetailsCard = ({ details }) => (
    <Card>
        <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
            <Database className="h-5 w-5" />
            Analysis Details
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-3">
                <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Frame Count:</span>
                    <span className="text-sm font-medium">{details.frameCount}</span>
                </div>
                <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Average Confidence:</span>
                    <span className="text-sm font-medium">{(details.avgConfidence * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Confidence Std Dev:</span>
                    <span className="text-sm font-medium">{(details.confidenceStd * 100).toFixed(1)}%</span>
                </div>
            </div>
            <div className="space-y-3">
                {details.temporalConsistency && (
                    <div className="flex justify-between">
                        <span className="text-sm text-gray-600">Temporal Consistency:</span>
                        <span className="text-sm font-medium">{(details.temporalConsistency * 100).toFixed(1)}%</span>
                    </div>
                )}
                {details.rollingAverage && (
                    <div className="flex justify-between">
                        <span className="text-sm text-gray-600">Rolling Average:</span>
                        <span className="text-sm font-medium">{(details.rollingAverage * 100).toFixed(1)}%</span>
                    </div>
                )}
            </div>
        </div>
    </Card>
);

// Frame Analysis Component
const FrameAnalysisCard = ({ frames }) => (
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
                        {frames.filter(f => f.prediction === "REAL").length}
                    </div>
                    <div className="text-sm text-gray-600">Real Frames</div>
                </div>
                <div className="text-center">
                    <div className="text-2xl font-bold text-red-600">
                        {frames.filter(f => f.prediction === "FAKE").length}
                    </div>
                    <div className="text-sm text-gray-600">Fake Frames</div>
                </div>
                <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">
                        {(frames.reduce((sum, f) => sum + f.confidence, 0) / frames.length * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-gray-600">Avg Confidence</div>
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
                                    <span className={`px-2 py-1 rounded text-xs ${
                                        frame.prediction === "REAL" ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"
                                    }`}>
                                        {frame.prediction}
                                    </span>
                                </td>
                                <td className="p-2">{(frame.confidence * 100).toFixed(1)}%</td>
                                <td className="p-2">{frame.timestamp ? `${frame.timestamp.toFixed(2)}s` : "N/A"}</td>
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

// Temporal Analysis Component
const TemporalAnalysisCard = ({ temporal }) => (
    <Card>
        <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Temporal Analysis
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
                <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Consistency Score:</span>
                    <span className="text-sm font-medium">{(temporal.consistencyScore * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Total Frames:</span>
                    <span className="text-sm font-medium">{temporal.totalFrames}</span>
                </div>
                <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Real Frames:</span>
                    <span className="text-sm font-medium text-green-600">{temporal.realFrames}</span>
                </div>
                <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Fake Frames:</span>
                    <span className="text-sm font-medium text-red-600">{temporal.fakeFrames}</span>
                </div>
            </div>
            <div className="space-y-4">
                <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Average Confidence:</span>
                    <span className="text-sm font-medium">{(temporal.avgConfidence * 100).toFixed(1)}%</span>
                </div>
                {temporal.patternDetection && (
                    <div>
                        <span className="text-sm text-gray-600">Pattern Detection:</span>
                        <p className="text-sm mt-1">{temporal.patternDetection}</p>
                    </div>
                )}
                {temporal.confidenceTrend && (
                    <div>
                        <span className="text-sm text-gray-600">Confidence Trend:</span>
                        <p className="text-sm mt-1">{temporal.confidenceTrend}</p>
                    </div>
                )}
                {temporal.anomalyFrames && temporal.anomalyFrames.length > 0 && (
                    <div>
                        <span className="text-sm text-gray-600">Anomaly Frames:</span>
                        <p className="text-sm mt-1">{temporal.anomalyFrames.join(", ")}</p>
                    </div>
                )}
            </div>
        </div>
    </Card>
);

// Model Info Component
const ModelInfoCard = ({ modelInfo }) => (
    <Card>
        <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
            <Cpu className="h-5 w-5" />
            Model Information
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-3">
                <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Version:</span>
                    <span className="text-sm font-medium">{modelInfo.version}</span>
                </div>
                <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Architecture:</span>
                    <span className="text-sm font-medium">{modelInfo.architecture}</span>
                </div>
                <div className="flex justify-between">
                    <span className="text-sm text-gray-600">Device:</span>
                    <span className="text-sm font-medium">{modelInfo.device}</span>
                </div>
            </div>
            <div className="space-y-3">
                {modelInfo.batchSize && (
                    <div className="flex justify-between">
                        <span className="text-sm text-gray-600">Batch Size:</span>
                        <span className="text-sm font-medium">{modelInfo.batchSize}</span>
                    </div>
                )}
                {modelInfo.numFrames && (
                    <div className="flex justify-between">
                        <span className="text-sm text-gray-600">Frames Processed:</span>
                        <span className="text-sm font-medium">{modelInfo.numFrames}</span>
                    </div>
                )}
            </div>
        </div>
    </Card>
);

// System Info Component
const SystemInfoCard = ({ systemInfo }) => (
    <Card>
        <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
            <Monitor className="h-5 w-5" />
            System Information
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-3">
                {systemInfo.processingDevice && (
                    <div className="flex justify-between">
                        <span className="text-sm text-gray-600">Processing Device:</span>
                        <span className="text-sm font-medium">{systemInfo.processingDevice}</span>
                    </div>
                )}
                {systemInfo.cudaAvailable !== undefined && (
                    <div className="flex justify-between">
                        <span className="text-sm text-gray-600">CUDA Available:</span>
                        <span className={`text-sm font-medium ${systemInfo.cudaAvailable ? 'text-green-600' : 'text-red-600'}`}>
                            {systemInfo.cudaAvailable ? 'Yes' : 'No'}
                        </span>
                    </div>
                )}
                {systemInfo.gpuMemoryUsed && (
                    <div className="flex justify-between">
                        <span className="text-sm text-gray-600">GPU Memory Used:</span>
                        <span className="text-sm font-medium">{systemInfo.gpuMemoryUsed}</span>
                    </div>
                )}
            </div>
            <div className="space-y-3">
                {systemInfo.systemMemoryUsed && (
                    <div className="flex justify-between">
                        <span className="text-sm text-gray-600">System Memory Used:</span>
                        <span className="text-sm font-medium">{systemInfo.systemMemoryUsed}</span>
                    </div>
                )}
                {systemInfo.loadBalancingInfo && (
                    <div>
                        <span className="text-sm text-gray-600">Load Balancing:</span>
                        <pre className="text-xs mt-1 p-2 bg-gray-100 dark:bg-gray-800 rounded">
                            {JSON.stringify(systemInfo.loadBalancingInfo, null, 2)}
                        </pre>
                    </div>
                )}
            </div>
        </div>
    </Card>
);

export default DetailedAnalysis;
