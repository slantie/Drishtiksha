// src/components/analysis/charts/video/VideoReport.jsx

import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "../../../ui/Card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "../../../ui/Tabs";
import { LineChart as LineChartIcon } from "lucide-react";

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

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="p-2 bg-gray-800 text-white rounded-md shadow-lg border border-gray-700 text-xs">
        <p className="font-bold">Frame: {label}</p>
        <p
          style={{
            color: data.prediction === "FAKE" ? "#ef4444" : "#22c56e",
          }}
        >
          Confidence: {data.confidence.toFixed(1)}% ({data.prediction})
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
      (f) => f.confidence * 100 >= i * 10 && f.confidence * 100 < i * 10 + 10
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
              <linearGradient id="colorConfidence" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f56565" stopOpacity={0.8} />
                <stop offset="95%" stopColor="#f56565" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="currentColor"
              className="opacity-15"
            />
            <XAxis dataKey="frame" tick={{ fontSize: 10 }} />
            <YAxis unit="%" domain={[0, 100]} tick={{ fontSize: 10 }} />
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
            <YAxis unit="%" domain={[0, 100]} tick={{ fontSize: 10 }} />
            <Tooltip
              content={<CustomTooltip />}
              cursor={{ fill: "rgba(128,128,128,0.1)" }}
            />
            <Bar dataKey="confidence">
              {chartData.map((e, i) => (
                <Cell
                  key={i}
                  fill={e.prediction === "FAKE" ? "#ef4444" : "#22c56e"}
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
            <YAxis unit="%" domain={[0, 100]} tick={{ fontSize: 10 }} />
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
            Frame {frame.frameNumber}: {(frame.confidence * 100).toFixed(1)}%
          </div>
        </div>
      ))}
    </div>
  );
};

export const VideoReport = ({ result }) => {
  const frames = result.frame_predictions || [];
  if (frames.length === 0)
    return <p>No frame-by-frame data available for this analysis.</p>;

  const realFrames = frames.filter((f) => f.prediction === "REAL").length;
  const fakeFrames = frames.length - realFrames;
  const avgScore = result.metrics?.final_average_score * 100 || 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <LineChartIcon className="h-5 w-5 text-primary-main" /> Frame-by-Frame
          Analysis
        </CardTitle>
        <CardDescription>
          Explore frame-level probability scores of potential deepfake content.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4 bg-light-muted-background dark:bg-dark-secondary rounded-lg mb-6 text-center">
          <div>
            <div className="text-2xl font-bold text-green-600">
              {realFrames}
            </div>
            <div className="text-xs">Authentic Frames</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-red-600">{fakeFrames}</div>
            <div className="text-xs">Deepfake Frames</div>
          </div>
          <div>
            <div className="text-2xl font-bold">{frames.length}</div>
            <div className="text-xs">Total Analyzed</div>
          </div>
          <div>
            <div
              className={`text-2xl font-bold ${
                avgScore > 50 ? "text-red-500" : "text-green-500"
              }`}
            >
              {avgScore.toFixed(1)}%
            </div>
            <div className="text-xs">Avg. Fake Score</div>
          </div>
        </div>
        <Tabs defaultValue="area">
          <TabsList className="grid w-full grid-cols-2 sm:grid-cols-4">
            <TabsTrigger value="area">Area Plot</TabsTrigger>
            <TabsTrigger value="bar">Bar Chart</TabsTrigger>
            <TabsTrigger value="trend">Trend Line</TabsTrigger>
            <TabsTrigger value="heatmap">Heatmap</TabsTrigger>
          </TabsList>
          <TabsContent value="area">
            <FrameAnalysisChart frames={frames} type="area" />
          </TabsContent>
          <TabsContent value="bar">
            <FrameAnalysisChart frames={frames} type="bar" />
          </TabsContent>
          <TabsContent value="trend">
            <FrameAnalysisChart frames={frames} type="trend" />
          </TabsContent>
          <TabsContent value="heatmap">
            <FrameAnalysisHeatmap frames={frames} />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};
