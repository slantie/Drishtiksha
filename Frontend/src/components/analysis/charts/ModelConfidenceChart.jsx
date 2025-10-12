// src/components/analysis/charts/ModelConfidenceChart.jsx
// Bar chart showing confidence levels for each model

import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { getModelColor } from "../../../utils/modelColors";

export const ModelConfidenceChart = ({ analyses }) => {
  if (!analyses || analyses.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-light-muted-text dark:text-dark-muted-text">
        No data available
      </div>
    );
  }

  // Transform data for chart
  const chartData = analyses.map((analysis) => ({
    name: analysis.modelName.replace(/-/g, " "),
    confidence: (analysis.confidence * 100).toFixed(2),
    prediction: analysis.prediction,
    fullName: analysis.modelName,
  }));

  // Sort by confidence descending
  chartData.sort((a, b) => b.confidence - a.confidence);

  // Get all model names for consistent color mapping
  const allModelNames = analyses.map((a) => a.modelName);

  return (
    <div>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={chartData}
          margin={{ top: 5, right: 30, left: -15, bottom: 5 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            className="stroke-gray-300 dark:stroke-gray-600"
          />
          <XAxis
            dataKey="name"
            axisLine={true}
            tickLine={true}
            tick={false}
            label="Models"
          />
          <YAxis
            domain={[0, 100]}
            axisLine={true}
            tickLine={true}
            tick={true}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "rgba(255, 255, 255, 0.95)",
              border: "1px solid #e2e8f0",
              borderRadius: "0.375rem",
            }}
            formatter={(value) => [`${value}%`, "Confidence"]}
            labelFormatter={(label, payload) => {
              if (payload && payload[0]) {
                return `${payload[0].payload.fullName}`;
              }
              return label;
            }}
          />
          <Bar dataKey="confidence" name="Confidence" radius={[8, 8, 0, 0]}>
            {chartData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={getModelColor(entry.fullName, allModelNames)}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Legend below the chart */}
      <div className="flex justify-center gap-4 flex-wrap">
        {chartData.map((item) => (
          <div key={item.fullName} className="flex items-center gap-2">
            <div
              className="w-4 h-4 rounded"
              style={{
                backgroundColor: getModelColor(item.fullName, allModelNames),
              }}
            ></div>
            <span className="text-sm">{item.name}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
