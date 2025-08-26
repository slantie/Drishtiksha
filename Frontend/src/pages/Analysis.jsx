// src/pages/Analysis.jsx

import React from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import {
    ArrowLeft,
    CheckCircle,
    AlertTriangle,
    RefreshCw,
    Cpu, // Used for Processing Environment
    Database, // Used for Analysis Metrics
    LineChart as LineChartIcon, // Used for Frame-by-Frame Analysis
    Video, // Used for the video player
    AlertCircle,
} from "lucide-react";
import { useMediaItemQuery } from "../hooks/useMediaQuery.jsx";
import { Button } from "../components/ui/Button";
import {
    Card,
    CardContent,
    CardHeader,
    CardTitle,
    CardDescription,
} from "../components/ui/Card";
import { PageLoader } from "../components/ui/LoadingSpinner";
import { MediaPlayer } from "../components/media/MediaPlayer.jsx";
import { MODEL_INFO } from "../constants/apiEndpoints.js";
import { PageHeader } from "../components/layout/PageHeader.jsx";
import { SkeletonCard } from "../components/ui/SkeletonCard.jsx";
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
    ComposedChart,
    Line,
    CartesianGrid,
    Tooltip,
    XAxis,
    YAxis,
    Cell,
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
                            data.prediction === "FAKE" ? "#ef4444" : "#22c56e",
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
                            <linearGradient
                                id="colorConfidence"
                                x1="0"
                                y1="0"
                                x2="0"
                                y2="1"
                            >
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
                        <CartesianGrid
                            strokeDasharray="3 3"
                            stroke="currentColor"
                            className="opacity-15"
                        />
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
                            fill="url(#colorConfidence)"
                        />
                    </AreaChart>
                );
            case "bar":
                return (
                    <BarChart
                        data={chartData}
                        margin={{ top: 5, right: 20, left: -20, bottom: 5 }}
                    >
                        <CartesianGrid
                            strokeDasharray="3 3"
                            stroke="currentColor"
                            className="opacity-15"
                        />
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
                                            : "#22c56e"
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
                        <CartesianGrid
                            strokeDasharray="3 3"
                            stroke="currentColor"
                            className="opacity-15"
                        />
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
                        <CartesianGrid
                            strokeDasharray="3 3"
                            stroke="currentColor"
                            className="opacity-15"
                        />
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
                                    fill={i < 5 ? "#22c56e" : "#ef4444"}
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
        <ResponsiveContainer width="100%" height={490}>
            {renderChart()}
        </ResponsiveContainer>
    );
};

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
            <div className="text-center items-center justify-center">
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
                <div className="w-full mt-3 h-10 flex items-center justify-center">
                    <p className="w-full max-w-md text-center">
                        {/* Explain the confidence score */}
                        The above confidence score represents how strongly the
                        model thinks it is a{" "}
                        <span className="inline font-bold text-lg">
                            {isReal ? "real" : "deepfake"}
                        </span>{" "}
                        video.
                    </p>
                </div>
            </div>
        </Card>
    );
};

const ProcessingEnvironmentCard = ({ modelInfo, systemInfo }) => {
    if (!modelInfo && !systemInfo) return null;
    return (
        <Card>
            <div className="p-6">
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                    <Cpu className="h-5 w-5 text-primary-main" />
                    Processing Environment
                </h3>
                <div className="grid grid-cols-1 gap-x-8 gap-y-4 text-sm">
                    {modelInfo && (
                        <>
                            <div className="flex justify-between">
                                <span>Model Name:</span>
                                <span className="font-medium font-sans">
                                    {modelInfo.modelName}
                                </span>
                            </div>
                            <div className="flex justify-between">
                                <span>Model Version:</span>
                                <span className="font-medium font-sans">
                                    {modelInfo.version}
                                </span>
                            </div>
                            <div className="flex justify-between">
                                <span>Architecture:</span>
                                <span className="font-medium font-sans">
                                    {modelInfo.architecture}
                                </span>
                            </div>
                            <div className="flex justify-between">
                                <span>Device:</span>
                                <span className="font-medium font-sans">
                                    {modelInfo.device}
                                </span>
                            </div>
                            {modelInfo.memoryUsage && (
                                <div className="flex justify-between">
                                    <span>Model Memory Usage:</span>
                                    <span className="font-medium font-sans">
                                        {modelInfo.memoryUsage}
                                    </span>
                                </div>
                            )}
                        </>
                    )}{" "}
                    {systemInfo && (
                        <>
                            <div className="flex justify-between">
                                <span>Processing Device:</span>
                                <span className="font-medium font-sans">
                                    {systemInfo.processingDevice}
                                </span>
                            </div>
                            {systemInfo.cudaAvailable !== undefined && (
                                <div className="flex justify-between">
                                    <span>CUDA Available:</span>
                                    <span
                                        className={`font-medium font-sans ${
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
                                    <span>GPU Memory Used:</span>
                                    <span className="font-medium font-sans">
                                        {systemInfo.gpuMemoryUsed}
                                    </span>
                                </div>
                            )}
                            {systemInfo.gpuMemoryTotal && (
                                <div className="flex justify-between">
                                    <span>GPU Memory Total:</span>
                                    <span className="font-medium font-sans">
                                        {systemInfo.gpuMemoryTotal}
                                    </span>
                                </div>
                            )}
                            {systemInfo.systemMemoryUsed && (
                                <div className="flex justify-between">
                                    <span>System Memory Used:</span>
                                    <span className="font-medium font-sans">
                                        {systemInfo.systemMemoryUsed}
                                    </span>
                                </div>
                            )}
                            {systemInfo.systemMemoryTotal && (
                                <div className="flex justify-between">
                                    <span>System Memory Total:</span>
                                    <span className="font-medium font-sans">
                                        {systemInfo.systemMemoryTotal}
                                    </span>
                                </div>
                            )}
                            {systemInfo.pythonVersion && (
                                <div className="flex justify-between">
                                    <span>Python Version:</span>
                                    <span className="font-medium font-sans">
                                        {systemInfo.pythonVersion}
                                    </span>
                                </div>
                            )}
                            {systemInfo.torchVersion && (
                                <div className="flex justify-between">
                                    <span>Torch Version:</span>
                                    <span className="font-medium font-sans">
                                        {systemInfo.torchVersion}
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
                            frame.prediction === "REAL" ? "#22c56e" : "#ef4444",
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

const AudioAnalysisReport = ({ analysis }) => {
    const { audioAnalysis } = analysis;
    if (!audioAnalysis) return null;

    const InfoItem = ({ label, value }) => (
        <div className="flex justify-between items-center border-b border-light-secondary/50 dark:border-dark-secondary/50 py-2">
            <span className="text-light-muted-text dark:text-dark-muted-text">{label}</span>
            <span className="font-semibold font-sans">{value}</span>
        </div>
    );
    
    return (
        <Card>
            <CardHeader><CardTitle className="flex items-center gap-2"><Music className="h-5 w-5 text-primary-main" /> Audio Forensic Report</CardTitle></CardHeader>
            <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                    <div>
                        <h3 className="font-semibold mb-2">Pitch Analysis</h3>
                        <InfoItem label="Mean Pitch (Hz)" value={audioAnalysis.meanPitchHz?.toFixed(2) || 'N/A'} />
                        <InfoItem label="Pitch Stability" value={audioAnalysis.pitchStabilityScore ? `${(audioAnalysis.pitchStabilityScore * 100).toFixed(1)}%` : 'N/A'} />
                    </div>
                    <div>
                        <h3 className="font-semibold mb-2">Energy Analysis</h3>
                        <InfoItem label="RMS Energy" value={audioAnalysis.rmsEnergy?.toFixed(4)} />
                        <InfoItem label="Silence Ratio" value={`${(audioAnalysis.silenceRatio * 100).toFixed(1)}%`} />
                    </div>
                    <div className="md:col-span-2">
                        <h3 className="font-semibold mb-2">Spectral Analysis</h3>
                        <InfoItem label="Spectral Centroid" value={audioAnalysis.spectralCentroid?.toFixed(2)} />
                        <InfoItem label="Spectral Contrast" value={audioAnalysis.spectralContrast?.toFixed(2)} />
                    </div>
                </div>
                {audioAnalysis.spectrogramUrl && (
                    <div className="pt-4">
                        <h3 className="font-semibold mb-2">Mel Spectrogram</h3>
                        <img src={audioAnalysis.spectrogramUrl} alt="Mel Spectrogram" className="rounded-lg border dark:border-gray-700 w-full" />
                    </div>
                )}
            </CardContent>
        </Card>
    )
}

const DetailedAnalysisSkeleton = () => (
    <div className="space-y-6">
        <SkeletonCard className="h-20" />
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-1 space-y-6">
                <SkeletonCard className="h-[400px]" />
                <SkeletonCard className="h-[250px]" />
            </div>
            <div className="lg:col-span-2 space-y-6">
                <SkeletonCard className="h-48" />
                <SkeletonCard className="h-64" />
                <SkeletonCard className="h-48" />
            </div>
        </div>
    </div>
);

const FrameAnalysisTabs = ({ analysis }) => {
    if (!analysis?.frameAnalysis?.length) return null;
    const { frameAnalysis: frames } = analysis;
    // Prefer pre-calculated stats from temporalAnalysis if available, otherwise calculate from frames
    const realFrames =
        analysis.temporalAnalysis?.realFrames ||
        frames.filter((f) => f.prediction === "REAL").length;
    const fakeFrames =
        analysis.temporalAnalysis?.fakeFrames ||
        frames.filter((f) => f.prediction === "FAKE").length;
    const avgConfidence =
        analysis.analysisDetails?.avgConfidence * 100 ||
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
                    Explore frame-level probability scores of fake content using
                    different visualizations.
                </CardDescription>
            </CardHeader>
            <CardContent>
                <div className="grid grid-cols-4 gap-4 p-4 bg-light-muted-background dark:bg-dark-secondary rounded-lg mb-6 text-center">
                    <div>
                        <div className="text-2xl font-bold text-green-600">
                            {realFrames}
                        </div>
                        <div className="text-xs text-light-muted-text dark:text-dark-muted-text">
                            Real Content Frames
                        </div>
                    </div>
                    <div>
                        <div className="text-2xl font-bold text-red-600">
                            {fakeFrames}
                        </div>
                        <div className="text-xs text-light-muted-text dark:text-dark-muted-text">
                            Fake Content Frames
                        </div>
                    </div>
                    <div>
                        <div className="text-2xl font-bold text-light-text dark:text-dark-text">
                            {realFrames + fakeFrames}
                        </div>
                        <div className="text-xs text-light-muted-text dark:text-dark-muted-text">
                            Total Frames
                        </div>
                    </div>
                    <div>
                        <div
                            className={`text-2xl font-bold ${
                                avgConfidence > 50
                                    ? "text-red-500"
                                    : "text-green-500"
                            }`}
                        >
                            {avgConfidence.toFixed(1)}%
                        </div>
                        <div className={`text-xs`}>
                            Fake Content Probability
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
    // UPDATED: Renamed URL params for clarity
    const { mediaId, analysisId } = useParams();
    const navigate = useNavigate();
    const {
        data: media,
        isLoading,
        error,
        refetch,
        isRefetching,
    } = useMediaItemQuery(mediaId);

    if (isLoading) return <DetailedAnalysisSkeleton />;

    if (error)
        return (
            <div className="text-center p-8">
                <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
                <h2 className="text-2xl font-bold mb-2">Error Loading Data</h2>
                <p className="mb-6">{error.message}</p>
                <Button onClick={() => navigate(`/results/${mediaId}`)}>
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
                <Button onClick={() => navigate(`/results/${mediaId}`)}>
                    <ArrowLeft className="mr-2 h-4 w-4" />
                    Back
                </Button>
            </div>
        );

    const modelLabel = MODEL_INFO[analysis.model]?.label || analysis.model;

    return (
        <div className="space-y-4">
            {/* <Breadcrumbs items={breadcrumbItems} /> */}
            <PageHeader
                title={`${modelLabel} Report`}
                description={`Forensic analysis details for ${video.filename}`}
                actions={
                    <>
                        <Button
                            onClick={() => {
                                refetch();
                                showToast.success(
                                    "Data refreshed successfully!"
                                );
                            }}
                            isLoading={isRefetching}
                            variant="outline"
                        >
                            <RefreshCw className="mr-2 h-4 w-4" /> Refresh Data
                        </Button>
                        <Button
                            onClick={() => {
                                navigate(-1);
                            }}
                            isLoading={isRefetching}
                            variant="outline"
                        >
                            <ArrowLeft className="mr-2 h-4 w-4" /> Go Back
                        </Button>
                    </>
                }
            />

            {/* Original Video Display - Moved to top for prominence */}
            <Card>
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <Video className="h-5 w-5 text-primary-main" /> Original
                        Video
                    </CardTitle>
                    <CardDescription>
                        Full view of the original video file.
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <MediaPlayer videoUrl={video.url} />
                </CardContent>
            </Card>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 items-start">
                <div className="lg:col-span-1 space-y-4 sticky top-24">
                    <AnalysisResultCard analysis={analysis} />
                    <ProcessingEnvironmentCard
                        modelInfo={analysis.modelInfo}
                        systemInfo={analysis.systemInfo}
                    />
                </div>
                <div className="lg:col-span-2 space-y-4">
                    <FrameAnalysisTabs analysis={analysis} />
                </div>
            </div>
        </div>
    );
};

export default Analysis;
