// src/components/analysis/charts/video/PredictionBarChart.jsx

import React from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  CartesianGrid,
  Tooltip,
  XAxis,
  YAxis,
  Cell,
} from "recharts";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "../../../ui/Card";
import { BarChart2 } from "lucide-react";

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
export const PredictionBarChart = ({ frames }) => {
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
          <BarChart2 className="h-5 w-5 text-primary-main" /> Frame Prediction
          Chart
        </CardTitle>
        <CardDescription>
          Each bar represents a frame, colored by its final prediction (REAL or
          FAKE).
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart
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
            <Tooltip
              content={<CustomTooltip />}
              cursor={{ fill: "rgba(128,128,128,0.1)" }}
            />
            <Bar dataKey="score">
              {chartData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.prediction === "FAKE" ? "#ef4444" : "#22c55e"}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};
