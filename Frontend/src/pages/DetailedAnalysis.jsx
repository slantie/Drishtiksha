// src/pages/DetailedAnalysis.jsx

import React, { useState } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import {
    ArrowLeft,
    Brain,
    CheckCircle,
    AlertTriangle,
    RefreshCw,
    TrendingUp,
    Cpu,
    Monitor,
    Database,
    LineChart,
    FileText,
    Loader2,
    AlertCircle,
} from "lucide-react";
import { useMediaItemQuery } from "../hooks/useMediaQuery.jsx";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { PageLoader } from "../components/ui/LoadingSpinner";
import { MediaPlayer } from "../components/media/MediaPlayer.jsx";
import { ANALYSIS_TYPE_INFO, MODEL_INFO } from "../constants/apiEndpoints.js";
import { showToast } from "../utils/toast.js";
import { useMediaMetadata } from "../hooks/useMediaMetadata.js";
import { SkeletonCard } from "../components/ui/SkeletonCard.jsx";

// FrameAnalysisCards Options

import {
    ScatterChart,
    Scatter,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ZAxis,
    ReferenceLine,
    Cell,
    ComposedChart,
    Area,
    AreaChart,
    Line,
    BarChart,
    Bar,
} from "recharts";
import { LineChart as LineChartIcon } from "lucide-react";

const FrameAnalysisCard1 = ({ frames }) => {
    if (!frames || frames.length === 0) return null;

    const realFrames = frames.filter((f) => f.prediction === "REAL").length;
    const fakeFrames = frames.length - realFrames;
    const avgConfidence =
        (frames.reduce((sum, f) => sum + f.confidence, 0) / frames.length) *
        100;

    const chartData = frames.map((frame) => ({
        frame: frame.frameNumber,
        confidence: frame.confidence * 100,
        prediction: frame.prediction,
    }));

    const CustomTooltip = ({ active, payload, label }) => {
        if (active && payload && payload.length) {
            return (
                <div className="p-2 bg-gray-800 text-white rounded-md shadow-lg border border-gray-700">
                    <p className="font-bold">{`Frame: ${label}`}</p>
                    <p
                        style={{
                            color:
                                payload[0].payload.prediction === "FAKE"
                                    ? "#ef4444"
                                    : "#22c55e",
                        }}
                    >
                        {`Confidence: ${payload[0].value.toFixed(1)}% (${
                            payload[0].payload.prediction
                        })`}
                    </p>
                </div>
            );
        }
        return null;
    };

    return (
        <Card>
            <div className="p-6">
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                    <LineChartIcon className="h-5 w-5 text-primary-main" />
                    Frame-by-Frame Confidence
                </h3>
                <div className="grid grid-cols-3 gap-4 p-4 bg-light-muted-background dark:bg-dark-muted-background rounded-lg mb-6">
                    <div className="text-center">
                        <div className="text-2xl font-bold text-green-600">
                            {realFrames}
                        </div>
                        <div className="text-xs text-gray-500">
                            Authentic Frames
                        </div>
                    </div>
                    <div className="text-center">
                        <div className="text-2xl font-bold text-red-600">
                            {fakeFrames}
                        </div>
                        <div className="text-xs text-gray-500">
                            Deepfake Frames
                        </div>
                    </div>
                    <div className="text-center">
                        <div className="text-2xl font-bold">
                            {avgConfidence.toFixed(1)}%
                        </div>
                        <div className="text-xs text-gray-500">
                            Avg. Confidence
                        </div>
                    </div>
                </div>

                <div style={{ width: "100%", height: 250 }}>
                    <ResponsiveContainer>
                        <AreaChart
                            data={chartData}
                            margin={{ top: 5, right: 20, left: -10, bottom: 5 }}
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
                                        stopColor="var(--color-primary)"
                                        stopOpacity={0.8}
                                    />
                                    <stop
                                        offset="95%"
                                        stopColor="var(--color-primary)"
                                        stopOpacity={0}
                                    />
                                </linearGradient>
                            </defs>
                            <CartesianGrid
                                strokeDasharray="3 3"
                                stroke="currentColor"
                                className="opacity-20"
                            />
                            <XAxis
                                dataKey="frame"
                                tick={{ fill: "currentColor", fontSize: 12 }}
                            />
                            <YAxis
                                domain={[0, 100]}
                                tick={{ fill: "currentColor", fontSize: 12 }}
                            />
                            <Tooltip content={<CustomTooltip />} />
                            <Area
                                type="monotone"
                                dataKey="confidence"
                                stroke="var(--color-primary)"
                                fillOpacity={1}
                                fill="url(#colorConfidence)"
                                strokeWidth={2}
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
                <style jsx global>{`
                    :root {
                        --color-primary: #f56565;
                    }
                    .dark {
                        --color-primary: #f56565;
                    }
                `}</style>
            </div>
        </Card>
    );
};

const FrameAnalysisCard2 = ({ frames }) => {
    const [tooltip, setTooltip] = useState(null);

    if (!frames || frames.length === 0) return null;

    const realFrames = frames.filter((f) => f.prediction === "REAL").length;
    const fakeFrames = frames.length - realFrames;
    const avgConfidence =
        (frames.reduce((sum, f) => sum + f.confidence, 0) / frames.length) *
        100;

    const svgWidth = 500;
    const svgHeight = 150;
    const points = frames.map((frame, i) => {
        const x = (i / (frames.length - 1)) * svgWidth;
        const y = svgHeight - frame.confidence * svgHeight;
        return { x, y, frame };
    });

    const pathD = points.reduce((acc, point, i, arr) => {
        if (i === 0) return `M ${point.x},${point.y}`;
        const prevPoint = arr[i - 1];
        const cp1 = { x: (prevPoint.x + point.x) / 2, y: prevPoint.y };
        const cp2 = { x: (prevPoint.x + point.x) / 2, y: point.y };
        return `${acc} C ${cp1.x},${cp1.y} ${cp2.x},${cp2.y} ${point.x},${point.y}`;
    }, "");

    const areaPathD = `${pathD} L ${svgWidth},${svgHeight} L 0,${svgHeight} Z`;

    const handleMouseMove = (e) => {
        const svg = e.currentTarget;
        const pt = svg.createSVGPoint();
        pt.x = e.clientX;
        const svgP = pt.matrixTransform(svg.getScreenCTM().inverse());
        const index = Math.round((svgP.x / svgWidth) * (points.length - 1));
        if (points[index]) {
            setTooltip(points[index]);
        }
    };

    return (
        <Card>
            <div className="p-6">
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                    <LineChartIcon className="h-5 w-5 text-primary-main" />
                    Frame-by-Frame Confidence
                </h3>
                <div className="grid grid-cols-3 gap-4 p-4 bg-light-muted-background dark:bg-dark-muted-background rounded-lg mb-6">
                    <div className="text-center">
                        <div className="text-2xl font-bold text-green-600">
                            {realFrames}
                        </div>
                        <div className="text-xs text-gray-500">
                            Authentic Frames
                        </div>
                    </div>
                    <div className="text-center">
                        <div className="text-2xl font-bold text-red-600">
                            {fakeFrames}
                        </div>
                        <div className="text-xs text-gray-500">
                            Deepfake Frames
                        </div>
                    </div>
                    <div className="text-center">
                        <div className="text-2xl font-bold">
                            {avgConfidence.toFixed(1)}%
                        </div>
                        <div className="text-xs text-gray-500">
                            Avg. Confidence
                        </div>
                    </div>
                </div>

                <div className="relative">
                    <svg
                        viewBox={`0 0 ${svgWidth} ${svgHeight}`}
                        className="w-full h-48"
                        onMouseMove={handleMouseMove}
                        onMouseLeave={() => setTooltip(null)}
                    >
                        <defs>
                            <linearGradient
                                id="svgAreaGradient"
                                x1="0"
                                y1="0"
                                x2="0"
                                y2="1"
                            >
                                <stop
                                    offset="0%"
                                    stopColor="#f56565"
                                    stopOpacity="0.4"
                                />
                                <stop
                                    offset="100%"
                                    stopColor="#f56565"
                                    stopOpacity="0"
                                />
                            </linearGradient>
                        </defs>
                        <path d={areaPathD} fill="url(#svgAreaGradient)" />
                        <path
                            d={pathD}
                            fill="none"
                            stroke="#f56565"
                            strokeWidth="2"
                        />
                        <line
                            x1="0"
                            y1={svgHeight / 2}
                            x2={svgWidth}
                            y2={svgHeight / 2}
                            stroke="currentColor"
                            strokeDasharray="3 3"
                            className="text-gray-400 opacity-50"
                        />
                        {tooltip && (
                            <circle
                                cx={tooltip.x}
                                cy={tooltip.y}
                                r="4"
                                fill="#f56565"
                                className="pointer-events-none"
                            />
                        )}
                    </svg>
                    {tooltip && (
                        <div
                            className="absolute p-2 text-xs bg-gray-800 text-white rounded-md pointer-events-none"
                            style={{
                                top: tooltip.y - 60,
                                left: tooltip.x,
                                transform: "translateX(-50%)",
                            }}
                        >
                            Frame {tooltip.frame.frameNumber}
                            <br />
                            Confidence:{" "}
                            {(tooltip.frame.confidence * 100).toFixed(1)}%
                        </div>
                    )}
                </div>
            </div>
        </Card>
    );
};

const FrameAnalysisCard3 = ({ frames }) => {
    if (!frames || frames.length === 0) return null;

    const realFrames = frames.filter((f) => f.prediction === "REAL").length;
    const fakeFrames = frames.length - realFrames;
    const avgConfidence =
        (frames.reduce((sum, f) => sum + f.confidence, 0) / frames.length) *
        100;

    return (
        <Card>
            <div className="p-6">
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                    <LineChartIcon className="h-5 w-5 text-primary-main" />
                    Frame Analysis Heatmap
                </h3>
                <div className="grid grid-cols-3 gap-4 p-4 bg-light-muted-background dark:bg-dark-muted-background rounded-lg mb-6">
                    <div className="text-center">
                        <div className="text-2xl font-bold text-green-600">
                            {realFrames}
                        </div>
                        <div className="text-xs text-gray-500">
                            Authentic Frames
                        </div>
                    </div>
                    <div className="text-center">
                        <div className="text-2xl font-bold text-red-600">
                            {fakeFrames}
                        </div>
                        <div className="text-xs text-gray-500">
                            Deepfake Frames
                        </div>
                    </div>
                    <div className="text-center">
                        <div className="text-2xl font-bold">
                            {avgConfidence.toFixed(1)}%
                        </div>
                        <div className="text-xs text-gray-500">
                            Avg. Confidence
                        </div>
                    </div>
                </div>

                <div className="w-full h-16 flex rounded-lg overflow-hidden border dark:border-gray-700">
                    {frames.map((frame) => (
                        <div
                            key={frame.frameNumber}
                            className="flex-1 group relative"
                            style={{
                                backgroundColor:
                                    frame.prediction === "REAL"
                                        ? "#22c55e"
                                        : "#ef4444",
                                opacity: 0.2 + frame.confidence * 0.8,
                            }}
                        >
                            <div className="absolute bottom-full mb-2 w-max p-2 text-xs bg-gray-800 text-white rounded-md opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none -translate-x-1/2 left-1/2">
                                Frame {frame.frameNumber}:{" "}
                                {(frame.confidence * 100).toFixed(1)}% (
                                {frame.prediction})
                            </div>
                        </div>
                    ))}
                </div>
                <div className="flex justify-between text-xs text-gray-500 mt-2 px-1">
                    <span>Start of Video</span>
                    <span>End of Video</span>
                </div>
            </div>
        </Card>
    );
};

const FrameAnalysisCard4 = ({ frames }) => {
    if (!frames || frames.length === 0) return null;

    const realFrames = frames.filter((f) => f.prediction === "REAL").length;
    const fakeFrames = frames.length - realFrames;
    const avgConfidence =
        (frames.reduce((sum, f) => sum + f.confidence, 0) / frames.length) *
        100;

    const chartData = frames.map((frame) => ({
        x: frame.frameNumber,
        y: frame.confidence * 100,
        prediction: frame.prediction,
    }));

    const CustomTooltip = ({ active, payload }) => {
        if (active && payload && payload.length) {
            const data = payload[0].payload;
            return (
                <div className="p-2 bg-gray-800 text-white rounded-md shadow-lg border border-gray-700">
                    <p className="font-bold">{`Frame: ${data.x}`}</p>
                    <p
                        style={{
                            color:
                                data.prediction === "FAKE"
                                    ? "#ef4444"
                                    : "#22c55e",
                        }}
                    >
                        {`Confidence: ${data.y.toFixed(1)}% (${
                            data.prediction
                        })`}
                    </p>
                </div>
            );
        }
        return null;
    };

    return (
        <Card>
            <div className="p-6">
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                    <LineChartIcon className="h-5 w-5 text-primary-main" />
                    Frame Confidence Plot
                </h3>
                <div className="grid grid-cols-3 gap-4 p-4 bg-light-muted-background dark:bg-dark-muted-background rounded-lg mb-6">
                    <div className="text-center">
                        <div className="text-2xl font-bold text-green-600">
                            {realFrames}
                        </div>
                        <div className="text-xs text-gray-500">
                            Authentic Frames
                        </div>
                    </div>
                    <div className="text-center">
                        <div className="text-2xl font-bold text-red-600">
                            {fakeFrames}
                        </div>
                        <div className="text-xs text-gray-500">
                            Deepfake Frames
                        </div>
                    </div>
                    <div className="text-center">
                        <div className="text-2xl font-bold">
                            {avgConfidence.toFixed(1)}%
                        </div>
                        <div className="text-xs text-gray-500">
                            Avg. Confidence
                        </div>
                    </div>
                </div>

                <div style={{ width: "100%", height: 250 }}>
                    <ResponsiveContainer>
                        <ScatterChart
                            margin={{ top: 5, right: 20, left: -10, bottom: 5 }}
                        >
                            <CartesianGrid
                                strokeDasharray="3 3"
                                stroke="currentColor"
                                className="opacity-20"
                            />
                            <XAxis
                                type="number"
                                dataKey="x"
                                name="Frame"
                                tick={{ fill: "currentColor", fontSize: 12 }}
                            />
                            <YAxis
                                type="number"
                                dataKey="y"
                                name="Confidence"
                                unit="%"
                                domain={[0, 100]}
                                tick={{ fill: "currentColor", fontSize: 12 }}
                            />
                            <ZAxis dataKey="prediction" name="prediction" />
                            <Tooltip
                                cursor={{ strokeDasharray: "3 3" }}
                                content={<CustomTooltip />}
                            />
                            <ReferenceLine
                                y={50}
                                label={{
                                    value: "50% Threshold",
                                    position: "insideTopLeft",
                                    fill: "currentColor",
                                    fontSize: 10,
                                    dy: -5,
                                }}
                                stroke="currentColor"
                                strokeDasharray="4 4"
                                className="opacity-50"
                            />
                            <Scatter
                                name="Frames"
                                data={chartData}
                                fill="#8884d8"
                            >
                                {chartData.map((entry, index) => (
                                    <Cell
                                        key={`cell-${index}`}
                                        fill={
                                            entry.prediction === "FAKE"
                                                ? "#ef4444"
                                                : "#22c55e"
                                        }
                                    />
                                ))}
                            </Scatter>
                        </ScatterChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </Card>
    );
};

const FrameAnalysisCard5 = ({ frames }) => {
    if (!frames || frames.length === 0) return null;

    const realFrames = frames.filter((f) => f.prediction === "REAL").length;
    const fakeFrames = frames.length - realFrames;
    const avgConfidence =
        (frames.reduce((sum, f) => sum + f.confidence, 0) / frames.length) *
        100;

    const chartData = frames.map((frame) => ({
        frame: frame.frameNumber,
        confidence: frame.confidence * 100,
        prediction: frame.prediction,
    }));

    const CustomTooltip = ({ active, payload, label }) => {
        if (active && payload && payload.length) {
            return (
                <div className="p-2 bg-gray-800 text-white rounded-md shadow-lg border border-gray-700">
                    <p className="font-bold">{`Frame: ${label}`}</p>
                    <p
                        style={{
                            color:
                                payload[0].payload.prediction === "FAKE"
                                    ? "#ef4444"
                                    : "#22c55e",
                        }}
                    >
                        {`Confidence: ${payload[0].value.toFixed(1)}% (${
                            payload[0].payload.prediction
                        })`}
                    </p>
                </div>
            );
        }
        return null;
    };

    return (
        <Card>
            <div className="p-6">
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                    <LineChartIcon className="h-5 w-5 text-primary-main" />
                    Frame Confidence Analysis
                </h3>
                <div className="grid grid-cols-3 gap-4 p-4 bg-light-muted-background dark:bg-dark-muted-background rounded-lg mb-6">
                    <div className="text-center">
                        <div className="text-2xl font-bold text-green-600">
                            {realFrames}
                        </div>
                        <div className="text-xs text-gray-500">
                            Authentic Frames
                        </div>
                    </div>
                    <div className="text-center">
                        <div className="text-2xl font-bold text-red-600">
                            {fakeFrames}
                        </div>
                        <div className="text-xs text-gray-500">
                            Deepfake Frames
                        </div>
                    </div>
                    <div className="text-center">
                        <div className="text-2xl font-bold">
                            {avgConfidence.toFixed(1)}%
                        </div>
                        <div className="text-xs text-gray-500">
                            Avg. Confidence
                        </div>
                    </div>
                </div>

                <div style={{ width: "100%", height: 250 }}>
                    <ResponsiveContainer>
                        <ComposedChart
                            data={chartData}
                            margin={{ top: 5, right: 20, left: -10, bottom: 5 }}
                        >
                            <defs>
                                <linearGradient
                                    id="hybridGradient"
                                    x1="0"
                                    y1="0"
                                    x2="0"
                                    y2="1"
                                >
                                    <stop
                                        offset="5%"
                                        stopColor="#f56565"
                                        stopOpacity={0.4}
                                    />
                                    <stop
                                        offset="95%"
                                        stopColor="#f56565"
                                        stopOpacity={0.1}
                                    />
                                </linearGradient>
                            </defs>
                            <CartesianGrid
                                strokeDasharray="3 3"
                                stroke="currentColor"
                                className="opacity-20"
                            />
                            <XAxis
                                dataKey="frame"
                                tick={{ fill: "currentColor", fontSize: 12 }}
                            />
                            <YAxis
                                domain={[0, 100]}
                                tick={{ fill: "currentColor", fontSize: 12 }}
                            />
                            <Tooltip content={<CustomTooltip />} />
                            <ReferenceLine
                                y={50}
                                stroke="currentColor"
                                strokeDasharray="4 4"
                                className="opacity-50"
                            />
                            <Area
                                type="monotone"
                                dataKey="confidence"
                                fill="url(#hybridGradient)"
                                stroke="none"
                            />
                            <Line
                                type="monotone"
                                dataKey="confidence"
                                stroke="#f56565"
                                strokeWidth={2}
                                dot={false}
                            />
                            {chartData.map((entry, index) => (
                                <Scatter
                                    key={`scatter-${index}`}
                                    dataKey="confidence"
                                    data={[{ ...entry }]}
                                    fill={
                                        entry.prediction === "FAKE"
                                            ? "#ef4444"
                                            : "#22c55e"
                                    }
                                    shape={<circle r={2} />}
                                />
                            ))}
                        </ComposedChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </Card>
    );
};

const FrameAnalysisCard6 = ({ frames, videoUrl }) => {
    const fps = useMediaMetadata(videoUrl, frames.length);

    if (!frames || frames.length === 0) return null;

    // const realFrames = frames.filter((f) => f.prediction === "REAL").length;
    // const fakeFrames = frames.length - realFrames;

    // Calculate rolling average for the trend line
    const rollingWindow = 10;
    const chartData = frames.map((frame, index) => {
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
            average: average,
            prediction: frame.prediction,
            time: (frame.frameNumber / fps).toFixed(2),
        };
    });

    const CustomTooltip = ({ active, payload, label }) => {
        if (active && payload && payload.length) {
            const data = payload[0].payload;
            return (
                <div className="p-2 bg-gray-800 text-white rounded-md shadow-lg border border-gray-700 text-xs">
                    <p className="font-bold">
                        Frame {label} (~{data.time}s)
                    </p>
                    <p>
                        Raw Confidence: {data.confidence.toFixed(1)}% (
                        {data.prediction})
                    </p>
                    <p>
                        Trend:{" "}
                        {payload
                            .find((p) => p.dataKey === "average")
                            ?.value.toFixed(1)}
                        %
                    </p>
                </div>
            );
        }
        return null;
    };

    return (
        <Card>
            <div className="p-6">
                <h3 className="text-lg font-bold mb-6 flex items-center gap-2">
                    <LineChartIcon className="h-5 w-5 text-primary-main" />
                    Frame Confidence Trend
                </h3>
                <div style={{ width: "100%", height: 250 }}>
                    <ResponsiveContainer>
                        <ComposedChart
                            data={chartData}
                            margin={{ top: 5, right: 20, left: -20, bottom: 5 }}
                        >
                            <CartesianGrid
                                strokeDasharray="3 3"
                                stroke="currentColor"
                                className="opacity-10"
                            />
                            <XAxis
                                dataKey="frame"
                                tick={{ fill: "currentColor", fontSize: 10 }}
                            />
                            <YAxis
                                yAxisId="left"
                                dataKey="confidence"
                                domain={[0, 100]}
                                tick={{ fill: "currentColor", fontSize: 10 }}
                            />
                            <YAxis
                                yAxisId="right"
                                dataKey="average"
                                orientation="right"
                                domain={[0, 100]}
                                tick={{ fill: "currentColor", fontSize: 10 }}
                            />
                            <Tooltip content={<CustomTooltip />} />
                            <ReferenceLine
                                y={50}
                                stroke="currentColor"
                                strokeDasharray="4 4"
                                className="opacity-50"
                            />
                            <Area
                                yAxisId="left"
                                type="monotone"
                                dataKey="confidence"
                                fill="#8884d8"
                                stroke="#8884d8"
                                fillOpacity={0.1}
                            />
                            <Line
                                yAxisId="right"
                                type="monotone"
                                dataKey="average"
                                stroke="#f56565"
                                strokeWidth={2}
                                dot={false}
                            />
                        </ComposedChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </Card>
    );
};

const FrameAnalysisCard7 = ({ frames }) => {
    if (!frames || frames.length === 0) return null;

    const bins = Array.from({ length: 10 }, (_, i) => i * 10); // 0, 10, 20...
    const histogramData = bins.map((binStart) => {
        const binEnd = binStart + 10;
        const count = frames.filter(
            (f) => f.confidence * 100 >= binStart && f.confidence * 100 < binEnd
        ).length;
        return {
            name: `${binStart}-${binEnd}%`,
            count: count,
        };
    });

    const CustomTooltip = ({ active, payload, label }) => {
        if (active && payload && payload.length) {
            return (
                <div className="p-2 bg-gray-800 text-white rounded-md shadow-lg border border-gray-700 text-xs">
                    <p className="font-bold">{`${payload[0].value} frames`}</p>
                    <p>{`Confidence: ${label}`}</p>
                </div>
            );
        }
        return null;
    };

    return (
        <Card>
            <div className="p-6">
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                    <LineChartIcon className="h-5 w-5 text-primary-main" />
                    Frame Confidence Distribution
                </h3>
                <p className="text-sm text-gray-500 mb-6">
                    This chart shows how many frames fall into each confidence
                    score bucket.
                </p>
                <div style={{ width: "100%", height: 250 }}>
                    <ResponsiveContainer>
                        <BarChart
                            data={histogramData}
                            margin={{ top: 5, right: 20, left: -10, bottom: 5 }}
                        >
                            <CartesianGrid
                                strokeDasharray="3 3"
                                stroke="currentColor"
                                className="opacity-10"
                            />
                            <XAxis
                                dataKey="name"
                                tick={{ fill: "currentColor", fontSize: 10 }}
                            />
                            <YAxis
                                tick={{ fill: "currentColor", fontSize: 10 }}
                                label={{
                                    value: "# of Frames",
                                    angle: -90,
                                    position: "insideLeft",
                                    fill: "currentColor",
                                    fontSize: 12,
                                }}
                            />
                            <Tooltip
                                content={<CustomTooltip />}
                                cursor={{ fill: "rgba(128,128,128,0.1)" }}
                            />
                            <Bar dataKey="count">
                                {histogramData.map((entry, index) => (
                                    <Cell
                                        key={`cell-${index}`}
                                        fill={index < 5 ? "#22c55e" : "#ef4444"}
                                        opacity={0.3 + index * 0.06}
                                    />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </Card>
    );
};

const FrameAnalysisCard8 = ({ frames, videoUrl }) => {
    const fps = useMediaMetadata(videoUrl, frames.length);
    const [hoverData, setHoverData] = useState(null);

    if (!frames || frames.length === 0) return null;
    const realFrames = frames.filter((f) => f.prediction === "REAL").length;
    const fakeFrames = frames.length - realFrames;
    const avgConfidence =
        (frames.reduce((sum, f) => sum + f.confidence, 0) / frames.length) *
        100;

    const svgWidth = 500;
    const svgHeight = 150;
    const points = frames.map((f, i) => [
        (i / (frames.length - 1)) * svgWidth,
        svgHeight - f.confidence * svgHeight,
    ]);
    const pathD = "M" + points.map((p) => `${p[0]},${p[1]}`).join(" L");

    const handleMouseMove = (e) => {
        const svg = e.currentTarget;
        const pt = svg.createSVGPoint();
        pt.x = e.clientX;
        const svgP = pt.matrixTransform(svg.getScreenCTM().inverse());

        const index = Math.round((svgP.x / svgWidth) * (frames.length - 1));
        const frame = frames[index];
        if (frame) {
            setHoverData({
                x: points[index][0],
                y: points[index][1],
                frameNum: frame.frameNumber,
                confidence: frame.confidence * 100,
                time: (frame.frameNumber / fps).toFixed(2),
            });
        }
    };

    return (
        <Card>
            <div className="p-6">
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                    <LineChartIcon className="h-5 w-5 text-primary-main" />
                    Frame Confidence Sparkline
                </h3>
                <div className="grid grid-cols-3 gap-4 p-4 bg-light-muted-background dark:bg-dark-muted-background rounded-lg mb-6">
                    <div className="text-center">
                        <div className="text-2xl font-bold text-green-600">
                            {realFrames}
                        </div>
                        <div className="text-xs text-gray-500">Authentic</div>
                    </div>
                    <div className="text-center">
                        <div className="text-2xl font-bold text-red-600">
                            {fakeFrames}
                        </div>
                        <div className="text-xs text-gray-500">Deepfake</div>
                    </div>
                    <div className="text-center">
                        <div className="text-2xl font-bold">
                            {avgConfidence.toFixed(1)}%
                        </div>
                        <div className="text-xs text-gray-500">
                            Avg. Confidence
                        </div>
                    </div>
                </div>
                <div className="relative">
                    <svg
                        viewBox={`0 0 ${svgWidth} ${svgHeight}`}
                        className="w-full h-48"
                        onMouseMove={handleMouseMove}
                        onMouseLeave={() => setHoverData(null)}
                    >
                        <path
                            d={pathD}
                            fill="none"
                            stroke="url(#lineGradient)"
                            strokeWidth="2"
                        />
                        <defs>
                            <linearGradient
                                id="lineGradient"
                                x1="0%"
                                y1="0%"
                                x2="100%"
                                y2="0%"
                            >
                                <stop offset="0%" stopColor="#22c55e" />
                                <stop offset="100%" stopColor="#ef4444" />
                            </linearGradient>
                        </defs>
                        {hoverData && (
                            <>
                                <line
                                    x1={hoverData.x}
                                    y1="0"
                                    x2={hoverData.x}
                                    y2={svgHeight}
                                    stroke="currentColor"
                                    strokeWidth="1"
                                    strokeDasharray="3 3"
                                    className="text-gray-500"
                                />
                                <circle
                                    cx={hoverData.x}
                                    cy={hoverData.y}
                                    r="4"
                                    fill="white"
                                    stroke="#f56565"
                                    strokeWidth="2"
                                />
                            </>
                        )}
                    </svg>
                    <div className="absolute top-0 right-0 p-2 bg-black/50 text-white text-xs rounded-bl-lg font-mono">
                        {hoverData
                            ? `Frame ${hoverData.frameNum} @ ${
                                  hoverData.time
                              }s: ${hoverData.confidence.toFixed(1)}%`
                            : `Hover for details`}
                    </div>
                </div>
            </div>
        </Card>
    );
};

// Page-specific sub-components for cleanliness and new features
const ReportHeader = ({
    modelName,
    analysisType,
    onRefresh,
    isRefetching,
    videoId,
}) => {
    const [isManuallyRefetching, setIsManuallyRefetching] = useState(false);

    const handleRefresh = async () => {
        setIsManuallyRefetching(true);
        try {
            await onRefresh();
            showToast.success("Report data has been updated.");
        } catch (error) {
            showToast.error("Failed to refresh data.");
            console.error("Error refreshing data:", error);
        } finally {
            setIsManuallyRefetching(false);
        }
    };

    return (
        <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
                <Link
                    to={`/results/${videoId}`}
                    aria-label="Back to results page"
                >
                    <Button variant="outline" size="sm">
                        <ArrowLeft className="h-4 w-4 mr-2" />
                        Back to Results
                    </Button>
                </Link>
                <div>
                    <h1 className="text-2xl font-bold">
                        {modelName} Forensic Report
                    </h1>
                    <p className="text-gray-500">
                        Analysis Type: {analysisType}
                    </p>
                </div>
            </div>
            <Button
                onClick={handleRefresh}
                variant="outline"
                size="sm"
                disabled={isRefetching || isManuallyRefetching}
                aria-label="Refresh analysis data"
            >
                {isRefetching || isManuallyRefetching ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                    <RefreshCw className="h-4 w-4 mr-2" />
                )}
                {isRefetching || isManuallyRefetching
                    ? "Refreshing..."
                    : "Refresh Data"}
            </Button>
        </div>
    );
};

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

// NEW: Component specifically for Audio Analysis data
const AudioAnalysisReport = ({ analysis }) => {
    const { pitch, energy, spectral, visualization } = analysis.audioAnalysis;
    return (
        <Card>
            <CardHeader>
                <CardTitle>Audio Forensic Details</CardTitle>
                <CardDescription>Detailed metrics extracted from the audio track.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <InfoCard title="Mean Pitch" value={pitch.meanPitchHz ? `${pitch.meanPitchHz.toFixed(1)} Hz` : 'N/A'} />
                    <InfoCard title="Pitch Stability" value={pitch.pitchStabilityScore ? `${(pitch.pitchStabilityScore * 100).toFixed(1)}%` : 'N/A'} />
                    <InfoCard title="RMS Energy" value={energy.rmsEnergy.toFixed(4)} />
                    <InfoCard title="Silence Ratio" value={`${(energy.silenceRatio * 100).toFixed(1)}%`} />
                    <InfoCard title="Spectral Centroid" value={spectral.spectralCentroid.toFixed(2)} />
                    <InfoCard title="Spectral Contrast" value={spectral.spectralContrast.toFixed(2)} />
                </div>
                {visualization.spectrogramUrl && (
                    <div>
                        <h4 className="font-semibold mb-2">Mel Spectrogram</h4>
                        <img src={visualization.spectrogramUrl} alt="Mel Spectrogram" className="rounded-lg border dark:border-gray-700 w-full" />
                    </div>
                )}
            </CardContent>
        </Card>
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

const DetailedAnalysis = () => {
    // UPDATED: Renamed URL params for clarity
    const { mediaId, analysisId } = useParams();
    const navigate = useNavigate();
    // UPDATED: Using the new generic hook and variable name
    const { data: media, isLoading, error, refetch, isRefetching } = useMediaItemQuery(mediaId);

    if (isLoading) return <AnalysisSkeleton />;
    if (error) return <div>Error: {error.message}</div>;

    const analysis = media?.analyses?.find((a) => a.id === analysisId);
    if (!media || !analysis) return <div>Analysis not found.</div>;

    const modelLabel = MODEL_INFO[analysis.model]?.label || analysis.model;

    return (
        <div className="space-y-4">
            <PageHeader
                title={`${modelLabel} Report`}
                description={`Forensic analysis details for ${media.filename}`}
                actions={<Button onClick={() => navigate(`/results/${mediaId}`)} variant="outline"><ArrowLeft className="mr-2 h-4 w-4"/>Back to Summary</Button>}
            />

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 items-start">
                <div className="lg:col-span-1 space-y-4 sticky top-24">
                    <Card>
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2"><Video className="h-5 w-5 text-primary-main" /> Original Media</CardTitle>
                        </CardHeader>
                        <CardContent>
                            {/* UPDATED: Using the generic MediaPlayer */}
                            <MediaPlayer media={media} />
                        </CardContent>
                    </Card>
                    <AnalysisResultCard analysis={analysis} />
                </div>
                <div className="lg:col-span-2 space-y-6">
                    {analysis.visualizedUrl && (
                        <Card>
                            <div className="p-6">
                                <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                                    <TrendingUp className="h-5 w-5 text-primary-main" />
                                    Analysis Visualization
                                </h3>
                                <MediaPlayer
                                    videoUrl={analysis.visualizedUrl}
                                />
                            </div>
                        </Card>
                    )}
                    {analysis.analysisDetails && (
                        <AnalysisDetailsCard
                            details={analysis.analysisDetails}
                        />
                    )}
                    {analysis.frameAnalysis &&
                        analysis.frameAnalysis.length > 0 && (
                            <div className="space-y-6">
                                {isRefetching ? (
                                    <SkeletonCard className="h-[400px]" />
                                ) : (
                                    analysis.frameAnalysis &&
                                    analysis.frameAnalysis.length > 0 && (
                                        <FrameAnalysisCard1
                                            frames={analysis.frameAnalysis}
                                        />
                                    )
                                )}
                                {isRefetching ? (
                                    <SkeletonCard className="h-[400px]" />
                                ) : (
                                    analysis.frameAnalysis &&
                                    analysis.frameAnalysis.length > 0 && (
                                        <FrameAnalysisCard2
                                            frames={analysis.frameAnalysis}
                                        />
                                    )
                                )}
                                {isRefetching ? (
                                    <SkeletonCard className="h-[400px]" />
                                ) : (
                                    analysis.frameAnalysis &&
                                    analysis.frameAnalysis.length > 0 && (
                                        <FrameAnalysisCard3
                                            frames={analysis.frameAnalysis}
                                        />
                                    )
                                )}
                                {isRefetching ? (
                                    <SkeletonCard className="h-[400px]" />
                                ) : (
                                    analysis.frameAnalysis &&
                                    analysis.frameAnalysis.length > 0 && (
                                        <FrameAnalysisCard4
                                            frames={analysis.frameAnalysis}
                                        />
                                    )
                                )}
                                {isRefetching ? (
                                    <SkeletonCard className="h-[400px]" />
                                ) : (
                                    analysis.frameAnalysis &&
                                    analysis.frameAnalysis.length > 0 && (
                                        <FrameAnalysisCard5
                                            frames={analysis.frameAnalysis}
                                        />
                                    )
                                )}
                                {isRefetching ? (
                                    <SkeletonCard className="h-[400px]" />
                                ) : (
                                    analysis.frameAnalysis &&
                                    analysis.frameAnalysis.length > 0 && (
                                        <FrameAnalysisCard6
                                            frames={analysis.frameAnalysis}
                                            videoUrl={video.url}
                                        />
                                    )
                                )}
                                {isRefetching ? (
                                    <SkeletonCard className="h-[400px]" />
                                ) : (
                                    analysis.frameAnalysis &&
                                    analysis.frameAnalysis.length > 0 && (
                                        <FrameAnalysisCard7
                                            frames={analysis.frameAnalysis}
                                        />
                                    )
                                )}
                                {isRefetching ? (
                                    <SkeletonCard className="h-[400px]" />
                                ) : (
                                    analysis.frameAnalysis &&
                                    analysis.frameAnalysis.length > 0 && (
                                        <FrameAnalysisCard8
                                            frames={analysis.frameAnalysis}
                                            videoUrl={video.url}
                                        />
                                    )
                                )}
                            </div>
                        )}
                    {(analysis.modelInfo || analysis.systemInfo) && (
                        <ProcessingEnvironmentCard
                            modelInfo={analysis.modelInfo}
                            systemInfo={analysis.systemInfo}
                        />
                    )}
                </div>
            </div>
        </div>
    );
};

export default DetailedAnalysis;
