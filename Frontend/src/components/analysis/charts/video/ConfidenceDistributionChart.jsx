// src/components/analysis/charts/video/ConfidenceDistributionChart.jsx

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
import { BarChartHorizontal, Activity } from "lucide-react";

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload?.length) {
    return (
      <div className="p-2 bg-gray-800 text-white rounded-md shadow-lg border border-gray-700 text-xs">
        <p className="font-bold">{payload[0].value} frames</p>
        <p>in {label} range</p>
      </div>
    );
  }
  return null;
};

export const ConfidenceDistributionChart = ({ frames }) => {
  if (!frames || frames.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChartHorizontal className="h-5 w-5 text-primary-main" /> Score
            Distribution
          </CardTitle>
          <CardDescription>
            Shows how many frames fall into each "fake" score bucket (0-100%).
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center p-8 text-light-muted-text dark:text-dark-muted-text">
            <Activity className="h-12 w-12 mx-auto mb-4" />
            <p className="text-lg font-semibold">
              No Frame Data for Distribution
            </p>
            <p className="mt-2 text-sm">
              Detailed frame analysis data is not available for this model.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Ensure score is converted to percentage for histogram bucketing
  const histogramData = Array.from({ length: 10 }, (_, i) => ({
    name: `${i * 10}-${i * 10 + 10}%`,
    count: frames.filter(
      (f) => f.score * 100 >= i * 10 && f.score * 100 < i * 10 + 10
    ).length,
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BarChartHorizontal className="h-5 w-5 text-primary-main" /> Score
          Distribution
        </CardTitle>
        <CardDescription>
          Shows how many frames fall into each "fake" score bucket (0-100%).
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart
            data={histogramData}
            margin={{ top: 5, right: 20, left: -20, bottom: 5 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="currentColor"
              className="opacity-15"
            />
            <XAxis dataKey="name" name="Score Range" tick={{ fontSize: 10 }} />
            <YAxis
              name="Frame Count"
              label={{
                value: "# Frames",
                angle: -90,
                position: "insideLeft",
                fill: "currentColor",
                fontSize: 12,
              }}
              tick={{ fontSize: 10 }}
            />
            <Tooltip
              content={<CustomTooltip />}
              cursor={{ fill: "rgba(128,128,128,0.1)" }}
            />
            <Bar dataKey="count">
              {histogramData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={index < 5 ? "#22c55e" : "#ef4444"} // Color based on range
                  opacity={0.5 + index * 0.05}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};
