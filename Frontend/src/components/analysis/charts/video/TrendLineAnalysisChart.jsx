// src/components/analysis/charts/video/TrendLineAnalysisChart.jsx

import React from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  CartesianGrid,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "../../../ui/Card";
import { TrendingUp, Activity } from "lucide-react";

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="p-2 bg-gray-800 text-white rounded-md shadow-lg border border-gray-700 text-xs">
        <p className="font-bold">Frame Index: {label}</p>
        <p
          style={{ color: data.prediction === "FAKE" ? "#ef4444" : "#22c55e" }}
        >
          Fake Score: {data.score.toFixed(1)}% ({data.prediction})
        </p>
        {data.average && <p>Trend: {data.average.toFixed(1)}%</p>}
      </div>
    );
  }
  return null;
};

export const TrendlineAnalysisChart = ({ frames }) => {
  if (!frames || frames.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-primary-main" /> Trendline
            Analysis
          </CardTitle>
          <CardDescription>
            Shows the raw frame scores (light area) with a smoothed trendline
            (red line) to highlight patterns.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center p-8 text-light-muted-text dark:text-dark-muted-text">
            <Activity className="h-12 w-12 mx-auto mb-4" />
            <p className="text-lg font-semibold">No Frame Data for Trendline</p>
            <p className="mt-2 text-sm">
              Detailed frame analysis data is not available for this model.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const chartData = frames.map((frame, index) => {
    const rollingWindow = Math.max(5, Math.floor(frames.length / 10)); // Dynamic window size
    const windowSlice = frames.slice(
      Math.max(0, index - rollingWindow + 1),
      index + 1
    );
    const average =
      windowSlice.reduce((sum, f) => sum + f.score * 100, 0) /
      windowSlice.length;
    return {
      index: frame.index,
      score: frame.score * 100,
      prediction: frame.prediction,
      average,
    };
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-primary-main" /> Trendline
          Analysis
        </CardTitle>
        <CardDescription>
          Shows the raw frame scores (light area) with a smoothed trendline (red
          line) to highlight patterns.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart
            data={chartData}
            margin={{ top: 5, right: 20, left: -20, bottom: 5 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="currentColor"
              className="opacity-15"
            />
            <XAxis dataKey="index" name="Frame Index" tick={{ fontSize: 10 }} />
            <YAxis unit="%" domain={[0, 100]} tick={{ fontSize: 10 }} />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey="score"
              fill="#8884d8"
              stroke="none"
              fillOpacity={0.2}
            />
            <Line
              type="monotone"
              dataKey="average"
              stroke="#ef4444"
              strokeWidth={2.5}
              dot={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};
