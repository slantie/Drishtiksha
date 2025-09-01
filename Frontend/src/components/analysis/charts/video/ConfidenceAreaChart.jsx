// src/components/analysis/charts/video/ConfidenceAreaChart.jsx

import React from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
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
import { AreaChart as AreaChartIcon } from "lucide-react";

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload?.length) {
    const data = payload[0].payload;
    return (
      <div className="p-2 bg-gray-800 text-white rounded-md shadow-lg border border-gray-700 text-xs">
        <p className="font-bold">Frame Index: {label}</p>
        <p
          style={{ color: data.prediction === "FAKE" ? "#ef4444" : "#22c55e" }}
        >
          Fake Score: {data.score.toFixed(1)}% ({data.prediction})
        </p>
      </div>
    );
  }
  return null;
};

export const ConfidenceAreaChart = ({ frames }) => {
  if (!frames || frames.length === 0) return null;

  const chartData = frames.map((frame) => ({
    index: frame.index,
    score: frame.score * 100,
    prediction: frame.prediction,
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <AreaChartIcon className="h-5 w-5 text-primary-main" /> Confidence
          Area Plot
        </CardTitle>
        <CardDescription>
          An overview of the "fake" probability score across all analyzed frames
          or sequences.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart
            data={chartData}
            margin={{ top: 5, right: 20, left: -20, bottom: 5 }}
          >
            <defs>
              <linearGradient
                id="confidenceGradient"
                x1="0"
                y1="0"
                x2="0"
                y2="1"
              >
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.7} />
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
              </linearGradient>
            </defs>
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
              stroke="#ef4444"
              strokeWidth={2}
              fill="url(#confidenceGradient)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};
