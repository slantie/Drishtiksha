// src/pages/Analysis.jsx

import React from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import {
    ArrowLeft,
    CheckCircle,
    AlertTriangle,
    RefreshCw,
    TrendingUp,
    Cpu,
    Database,
    FileText,
    LineChart as LineChartIcon,
    Video,
    AlertCircle,
} from "lucide-react";
import { useVideoQuery } from "../hooks/useVideosQuery.jsx";
import { Button } from "../components/ui/Button";
import {
    Card,
    CardContent,
    CardHeader,
    CardTitle,
    CardDescription,
} from "../components/ui/Card";
import { PageLoader } from "../components/ui/LoadingSpinner";
import { VideoPlayer } from "../components/videos/VideoPlayer.jsx";
import { MODEL_INFO } from "../constants/apiEndpoints.js";
import { PageHeader } from "../components/layout/PageHeader.jsx";
// import { Breadcrumbs } from "../components/ui/Breadcrumbs.jsx";
import {
    Tabs,
    TabsList,
    TabsTrigger,
    TabsContent,
} from "../components/ui/Tabs.jsx";
import {
    ResponsiveContainer,
    AreaChart,
    Area,
    BarChart,
    Bar,
    ScatterChart,
    Scatter,
    ComposedChart,
    Line,
    CartesianGrid,
    Tooltip,
    XAxis,
    YAxis,
    Cell,
    ReferenceLine,
} from "recharts";
import showToast from "../utils/toast.js";

// --- SUB-COMPONENTS ---

const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
        const data = payload[0].payload;
        return (
            <div className="p-2 bg-gray-800 text-white rounded-md shadow-lg border border-gray-700 text-xs">
                <p className="font-bold">Frame: {label}</p>
                <p
                    style={{
                        color:
                            data.prediction === "FAKE" ? "#ef4444" : "#22c55e",
                    }}
                >
                    Confidence: {data.confidence.toFixed(1)}% ({data.prediction}
                    )
                </p>
                {data.average && <p>Trend: {data.average.toFixed(1)}%</p>}
            </div>
        );
    }
    return null;
};

// REFACTOR: This single component now handles multiple Recharts visualizations.
const FrameAnalysisChart = ({ frames, type }) => {
    if (!frames || frames.length === 0) return null;
    const chartData = frames.map((frame, index) => {
        const rollingWindow = 10;
        const windowSlice = frames.slice(
            Math.max(0, index - rollingWindow + 1),
            index + 1
        );
        const average =
            windowSlice.reduce((sum, f) => sum + f.confidence * 100, 0) /
            windowSlice.length;
        return {
            frame: frame.frameNumber,
            confidence: frame.confidence * 100,
            prediction: frame.prediction,
            average,
        };
    });
    const histogramData = Array.from({ length: 10 }, (_, i) => ({
        name: `${i * 10}-${i * 10 + 10}%`,
        count: frames.filter(
            (f) =>
                f.confidence * 100 >= i * 10 && f.confidence * 100 < i * 10 + 10
        ).length,
    }));

    const renderChart = () => {
        switch (type) {
            case "area":
                return (
                    <AreaChart
                        data={chartData}
                        margin={{ top: 5, right: 20, left: -20, bottom: 5 }}
                    >
                        <defs>
                            <linearGradient id="a" x1="0" y1="0" x2="0" y2="1">
                                <stop
                                    offset="5%"
                                    stopColor="#f56565"
                                    stopOpacity={0.8}
                                />
                                <stop
                                    offset="95%"
                                    stopColor="#f56565"
                                    stopOpacity={0}
                                />
                            </linearGradient>
                        </defs>
                        <CartesianGrid />
                        <XAxis dataKey="frame" tick={{ fontSize: 10 }} />
                        <YAxis
                            unit="%"
                            domain={[0, 100]}
                            tick={{ fontSize: 10 }}
                        />
                        <Tooltip content={<CustomTooltip />} />
                        <Area
                            type="monotone"
                            dataKey="confidence"
                            stroke="#f56565"
                            fill="url(#a)"
                        />
                    </AreaChart>
                );
            case "bar":
                return (
                    <BarChart
                        data={chartData}
                        margin={{ top: 5, right: 20, left: -20, bottom: 5 }}
                    >
                        <CartesianGrid />
                        <XAxis dataKey="frame" tick={{ fontSize: 10 }} />
                        <YAxis
                            unit="%"
                            domain={[0, 100]}
                            tick={{ fontSize: 10 }}
                        />
                        <Tooltip
                            content={<CustomTooltip />}
                            cursor={{ fill: "rgba(128,128,128,0.1)" }}
                        />
                        <Bar dataKey="confidence">
                            {chartData.map((e, i) => (
                                <Cell
                                    key={i}
                                    fill={
                                        e.prediction === "FAKE"
                                            ? "#ef4444"
                                            : "#22c55e"
                                    }
                                />
                            ))}
                        </Bar>
                    </BarChart>
                );
            case "trend":
                return (
                    <ComposedChart
                        data={chartData}
                        margin={{ top: 5, right: 20, left: -20, bottom: 5 }}
                    >
                        <CartesianGrid />
                        <XAxis dataKey="frame" tick={{ fontSize: 10 }} />
                        <YAxis
                            unit="%"
                            domain={[0, 100]}
                            tick={{ fontSize: 10 }}
                        />
                        <Tooltip content={<CustomTooltip />} />
                        <Area
                            type="monotone"
                            dataKey="confidence"
                            fill="#8884d8"
                            stroke="#8884d8"
                            fillOpacity={0.1}
                        />
                        <Line
                            type="monotone"
                            dataKey="average"
                            stroke="#f56565"
                            strokeWidth={2}
                            dot={false}
                        />
                    </ComposedChart>
                );
            case "distribution":
                return (
                    <BarChart
                        data={histogramData}
                        margin={{ top: 5, right: 20, left: -20, bottom: 5 }}
                    >
                        <CartesianGrid />
                        <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                        <YAxis
                            label={{
                                value: "# Frames",
                                angle: -90,
                                position: "insideLeft",
                                fill: "currentColor",
                                fontSize: 12,
                            }}
                            tick={{ fontSize: 10 }}
                        />
                        <Tooltip cursor={{ fill: "rgba(128,128,128,0.1)" }} />
                        <Bar dataKey="count">
                            {histogramData.map((e, i) => (
                                <Cell
                                    key={i}
                                    fill={i < 5 ? "#22c55e" : "#ef4444"}
                                    opacity={0.4 + i * 0.06}
                                />
                            ))}
                        </Bar>
                    </BarChart>
                );
            default:
                return null;
        }
    };
    return (
        <ResponsiveContainer width="100%" height={250}>
            {renderChart()}
        </ResponsiveContainer>
    );
};

// Other helper components
const AnalysisResultCard = ({ analysis }) => {
    if (!analysis) return null;
    const isReal = analysis.prediction === "REAL";
    const confidence = analysis.confidence * 100;
    return (
        <Card
            className={`border-2 p-6 ${
                isReal ? "border-green-500/30" : "border-red-500/30"
            }`}
            padding="lg"
        >
            <div className="text-center">
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
        </Card>
    );
};
const AnalysisDetailsCard = ({ details }) =>
    !details ? null : (
        <Card>
            <div className="p-6">
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                    <Database className="h-5 w-5 text-primary-main" />
                    Analysis Metrics
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-4 text-sm">
                    <div className="flex justify-between">
                        <span>Frame Count:</span>
                        <span className="font-medium font-mono">
                            {details.frameCount}
                        </span>
                    </div>
                    <div className="flex justify-between">
                        <span>Avg. Confidence:</span>
                        <span className="font-medium font-mono">
                            {(details.avgConfidence * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div className="flex justify-between">
                        <span>Confidence Std Dev:</span>
                        <span className="font-medium font-mono">
                            {(details.confidenceStd * 100).toFixed(1)}%
                        </span>
                    </div>
                    {details.temporalConsistency && (
                        <div className="flex justify-between">
                            <span>Temporal Consistency:</span>
                            <span className="font-medium font-mono">
                                {(details.temporalConsistency * 100).toFixed(1)}
                                %
                            </span>
                        </div>
                    )}
                </div>
            </div>
        </Card>
    );
const ProcessingEnvironmentCard = ({ modelInfo, systemInfo }) => {
    if (!modelInfo && !systemInfo) return null;
    return (
        <Card>
            <div className="p-6">
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                    <Cpu className="h-5 w-5 text-primary-main" />
                    Processing Environment
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-4 text-sm">
                    {modelInfo && (
                        <>
                            <div className="flex justify-between">
                                <span>Model Version:</span>
                                <span className="font-medium font-mono">
                                    {modelInfo.version}
                                </span>
                            </div>
                            <div className="flex justify-between">
                                <span>Architecture:</span>
                                <span className="font-medium font-mono">
                                    {modelInfo.architecture}
                                </span>
                            </div>
                        </>
                    )}{" "}
                    {systemInfo && (
                        <>
                            <div className="flex justify-between">
                                <span>Processing Device:</span>
                                <span className="font-medium font-mono">
                                    {systemInfo.processingDevice}
                                </span>
                            </div>
                            {systemInfo.cudaAvailable !== undefined && (
                                <div className="flex justify-between">
                                    <span>CUDA Available:</span>
                                    <span
                                        className={`font-medium font-mono ${
                                            systemInfo.cudaAvailable
                                                ? "text-green-600"
                                                : "text-red-600"
                                        }`}
                                    >
                                        {systemInfo.cudaAvailable
                                            ? "Yes"
                                            : "No"}
                                    </span>
                                </div>
                            )}
                            {systemInfo.gpuMemoryUsed && (
                                <div className="flex justify-between">
                                    <span>GPU Memory:</span>
                                    <span className="font-medium font-mono">
                                        {systemInfo.gpuMemoryUsed}
                                    </span>
                                </div>
                            )}
                            {systemInfo.systemMemoryUsed && (
                                <div className="flex justify-between">
                                    <span>System Memory:</span>
                                    <span className="font-medium font-mono">
                                        {systemInfo.systemMemoryUsed}
                                    </span>
                                </div>
                            )}
                        </>
                    )}
                </div>
            </div>
        </Card>
    );
};

const FrameAnalysisHeatmap = ({ frames }) => {
    if (!frames || frames.length === 0) return null;
    return (
        <div className="w-full h-16 flex rounded-lg overflow-hidden border dark:border-gray-700">
            {frames.map((frame) => (
                <div
                    key={frame.frameNumber}
                    className="flex-1 group relative"
                    style={{
                        backgroundColor:
                            frame.prediction === "REAL" ? "#22c55e" : "#ef4444",
                        opacity: 0.2 + frame.confidence * 0.8,
                    }}
                >
                    <div className="absolute bottom-full mb-2 w-max p-2 text-xs bg-gray-800 text-white rounded-md opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none -translate-x-1/2 left-1/2 z-10">
                        Frame {frame.frameNumber}:{" "}
                        {(frame.confidence * 100).toFixed(1)}%
                    </div>
                </div>
            ))}
        </div>
    );
};

// REFACTOR: This new component encapsulates the summary stats and the tabs for a cleaner main component.
const FrameAnalysisTabs = ({ analysis }) => {
    if (!analysis?.frameAnalysis?.length) return null;
    const { frameAnalysis: frames } = analysis;
    const realFrames = frames.filter((f) => f.prediction === "REAL").length;
    const fakeFrames = frames.length - realFrames;
    const avgConfidence =
        (frames.reduce((sum, f) => sum + f.confidence, 0) / frames.length) *
        100;

    return (
        <Card>
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <LineChartIcon className="h-5 w-5 text-primary-main" />{" "}
                    Frame-by-Frame Analysis
                </CardTitle>
                <CardDescription>
                    Explore frame-level confidence scores using different
                    visualizations.
                </CardDescription>
            </CardHeader>
            <CardContent>
                <div className="grid grid-cols-3 gap-4 p-4 bg-light-muted-background dark:bg-dark-secondary rounded-lg mb-6 text-center">
                    <div>
                        <div className="text-2xl font-bold text-green-600">
                            {realFrames}
                        </div>
                        <div className="text-xs text-light-muted-text dark:text-dark-muted-text">
                            Authentic Frames
                        </div>
                    </div>
                    <div>
                        <div className="text-2xl font-bold text-red-600">
                            {fakeFrames}
                        </div>
                        <div className="text-xs text-light-muted-text dark:text-dark-muted-text">
                            Deepfake Frames
                        </div>
                    </div>
                    <div>
                        <div className="text-2xl font-bold">
                            {avgConfidence.toFixed(1)}%
                        </div>
                        <div className="text-xs text-light-muted-text dark:text-dark-muted-text">
                            Avg. Confidence
                        </div>
                    </div>
                </div>
                <Tabs defaultValue="area">
                    <TabsList className="grid w-full grid-cols-3 sm:grid-cols-5">
                        <TabsTrigger value="area">Area Plot</TabsTrigger>
                        <TabsTrigger value="bar">Bar Chart</TabsTrigger>
                        <TabsTrigger value="trend">Trend Line</TabsTrigger>
                        <TabsTrigger value="distribution">
                            Distribution
                        </TabsTrigger>
                        <TabsTrigger value="heatmap">Heatmap</TabsTrigger>
                    </TabsList>
                    <TabsContent value="area" className="mt-4">
                        <FrameAnalysisChart frames={frames} type="area" />
                    </TabsContent>
                    <TabsContent value="bar" className="mt-4">
                        <FrameAnalysisChart frames={frames} type="bar" />
                    </TabsContent>
                    <TabsContent value="trend" className="mt-4">
                        <FrameAnalysisChart frames={frames} type="trend" />
                    </TabsContent>
                    <TabsContent value="distribution" className="mt-4">
                        <FrameAnalysisChart
                            frames={frames}
                            type="distribution"
                        />
                    </TabsContent>
                    <TabsContent value="heatmap" className="mt-4 pt-4">
                        <FrameAnalysisHeatmap frames={frames} />
                    </TabsContent>
                </Tabs>
            </CardContent>
        </Card>
    );
};

const Analysis = () => {
    const { videoId, analysisId } = useParams();
    const navigate = useNavigate();
    const {
        data: video,
        isLoading,
        error,
        refetch,
        isRefetching,
    } = useVideoQuery(videoId);

    if (isLoading) return <DetailedAnalysisSkeleton />;

    if (error)
        return (
            <div className="text-center p-8">
                <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
                <h2 className="text-2xl font-bold mb-2">Error Loading Data</h2>
                <p className="mb-6">{error.message}</p>
                <Button onClick={() => navigate(`/results/${videoId}`)}>
                    <ArrowLeft className="mr-2 h-4 w-4" />
                    Back
                </Button>
            </div>
        );

    const analysis = video?.analyses?.find((a) => a.id === analysisId);
    if (!video || !analysis)
        return (
            <div className="text-center p-8">
                <AlertTriangle className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <h2 className="text-2xl font-bold mb-2">Analysis Not Found</h2>
                <p className="mb-6">
                    The requested analysis could not be found.
                </p>
                <Button onClick={() => navigate(`/results/${videoId}`)}>
                    <ArrowLeft className="mr-2 h-4 w-4" />
                    Back
                </Button>
            </div>
        );

    const modelInfo = MODEL_INFO[analysis.model];

    return (
        <div className="space-y-4">
            {/* <Breadcrumbs items={breadcrumbItems} /> */}
            <PageHeader
                title={`${modelInfo?.label || analysis.model} Report`}
                description={`Forensic analysis details for ${video.filename}`}
                actions={
                    <Button
                        onClick={() => {
                            refetch();
                            showToast.success("Data refreshed successfully!");
                        }}
                        isLoading={isRefetching}
                        variant="outline"
                    >
                        <RefreshCw className="mr-2 h-4 w-4" /> Refresh Data
                    </Button>
                }
            />

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 items-start">
                <div className="lg:col-span-1 space-y-4 sticky top-24">
                    <Card>
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                                <Video className="h-5 w-5 text-primary-main" />{" "}
                                Original Video
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <VideoPlayer videoUrl={video.url} />
                        </CardContent>
                    </Card>
                    <AnalysisResultCard analysis={analysis} />
                    {analysis.analysisDetails && (
                        <ProcessingEnvironmentCard
                            title="Analysis Metrics"
                            icon={Database}
                            data={[
                                {
                                    label: "Frame Count",
                                    value: analysis.analysisDetails.frameCount,
                                },
                                {
                                    label: "Avg. Confidence",
                                    value: `${(
                                        analysis.analysisDetails.avgConfidence *
                                        100
                                    ).toFixed(1)}%`,
                                },
                            ]}
                        />
                    )}
                    {analysis.systemInfo && (
                        <ProcessingEnvironmentCard
                            title="Processing Environment"
                            icon={Cpu}
                            data={[
                                {
                                    label: "Device",
                                    value: analysis.systemInfo.processingDevice,
                                },
                            ]}
                        />
                    )}
                </div>
                <div className="lg:col-span-2 space-y-4">
                    {analysis.visualizedUrl && (
                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <TrendingUp className="h-5 w-5 text-primary-main" />{" "}
                                    Analysis Visualization
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <VideoPlayer
                                    videoUrl={analysis.visualizedUrl}
                                />
                            </CardContent>
                        </Card>
                    )}
                    <FrameAnalysisTabs analysis={analysis} />
                </div>
            </div>
        </div>
    );
};

export default Analysis;
