// src/components/analysis/charts/FrameAnalysisChart.jsx
// Line chart showing frame-by-frame confidence trends for all models combined

import React, { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { getModelColor } from "../../../utils/modelColors";

export const FrameAnalysisChart = ({ analyses }) => {
  // Extract analyses that have frame predictions
  const analysesWithFrames = useMemo(() => {
    return (
      analyses?.filter(
        (a) =>
          a.resultPayload?.frame_predictions &&
          a.resultPayload.frame_predictions.length > 0
      ) || []
    );
  }, [analyses]);

  // Get all model names for consistent color mapping
  const allModelNames = useMemo(() => {
    return analyses?.map((a) => a.modelName) || [];
  }, [analyses]);

  // Process and normalize all model data
  const chartData = useMemo(() => {
    if (analysesWithFrames.length === 0) return [];

    // Find the maximum number of data points across all models
    const maxDataPoints = Math.max(
      ...analysesWithFrames.map((a) => a.resultPayload.frame_predictions.length)
    );

    // Determine a reasonable number of points to display (max 100)
    const targetPoints = Math.min(maxDataPoints, 100);

    // Create normalized data structure
    const normalizedData = [];

    for (let i = 0; i < targetPoints; i++) {
      const dataPoint = {
        index: i,
        position: ((i / (targetPoints - 1)) * 100).toFixed(1), // Percentage through video
      };

      // Add data for each model, interpolating as needed
      analysesWithFrames.forEach((analysis) => {
        const predictions = analysis.resultPayload.frame_predictions;
        const modelName = analysis.modelName;

        // Normalize: map the current point to the model's data range
        const normalizedIndex =
          (i / (targetPoints - 1)) * (predictions.length - 1);

        // Interpolate between two nearest points
        const lowerIndex = Math.floor(normalizedIndex);
        const upperIndex = Math.ceil(normalizedIndex);
        const fraction = normalizedIndex - lowerIndex;

        let score;
        if (lowerIndex === upperIndex) {
          // Exact match
          score = predictions[lowerIndex].score * 100;
        } else {
          // Linear interpolation
          const lowerScore = predictions[lowerIndex].score * 100;
          const upperScore = predictions[upperIndex].score * 100;
          score = lowerScore + (upperScore - lowerScore) * fraction;
        }

        dataPoint[modelName] = score.toFixed(2);
      });

      normalizedData.push(dataPoint);
    }

    return normalizedData;
  }, [analysesWithFrames]);

  if (analysesWithFrames.length === 0) {
    return (
      <div className="h-64 flex flex-col items-center justify-center text-light-muted-text dark:text-dark-muted-text">
        <p>No frame-level analysis data available</p>
        <p className="text-sm mt-2">
          Some models don't provide frame-by-frame predictions
        </p>
      </div>
    );
  }

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white dark:bg-gray-800 p-3 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
          <p className="font-semibold text-sm mb-2">Position: {label}%</p>
          {payload.map((entry, index) => (
            <div key={index} className="flex items-center gap-2 text-xs">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: entry.color }}
              />
              <span className="font-medium">{entry.name}:</span>
              <span>{entry.value}%</span>
            </div>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-4">
      {/* Chart */}
      <ResponsiveContainer width="100%" height={400}>
        <LineChart
          data={chartData}
          margin={{ top: 5, right: 30, left: 0, bottom: 5 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            className="stroke-gray-300 dark:stroke-gray-600"
          />
          <XAxis
            dataKey="position"
            label={{
              value: "Position in Video (%)",
              position: "insideBottom",
              offset: -15,
            }}
            tick={{ fill: "currentColor", fontSize: 12 }}
            tickFormatter={(value) => `${value}%`}
          />
          <YAxis
            domain={[0, 100]}
            tick={{ fill: "currentColor", fontSize: 12 }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: "12px", paddingTop: "20px" }}
            formatter={(value) => value.replace(/-/g, " ")}
          />

          {/* Render a line for each model */}
          {analysesWithFrames.map((analysis) => (
            <Line
              key={analysis.id}
              type="monotone"
              dataKey={analysis.modelName}
              name={analysis.modelName}
              stroke={getModelColor(analysis.modelName, allModelNames)}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};
