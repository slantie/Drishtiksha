// src/components/analysis/charts/video/FrameAnalysisChart.jsx

import React from "react";
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

export const FrameAnalysisChart = ({ frames, type }) => {
  if (!frames || frames.length === 0) return null;

  const chartData = frames.map((frame, index) => {
    const rollingWindow = 10;
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

  const histogramData = Array.from({ length: 10 }, (_, i) => ({
    name: `${i * 10}-${i * 10 + 10}%`,
    count: frames.filter(
      (f) => f.score * 100 >= i * 10 && f.score * 100 < i * 10 + 10
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
              <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8} />
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="currentColor"
              className="opacity-15"
            />
            <XAxis dataKey="index" tick={{ fontSize: 10 }} />
            <YAxis unit="%" domain={[0, 100]} tick={{ fontSize: 10 }} />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey="score"
              stroke="#ef4444"
              fill="url(#colorScore)"
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
            <XAxis dataKey="index" tick={{ fontSize: 10 }} />
            <YAxis unit="%" domain={[0, 100]} tick={{ fontSize: 10 }} />
            <Tooltip
              content={<CustomTooltip />}
              cursor={{ fill: "rgba(128,128,128,0.1)" }}
            />
            <Bar dataKey="score">
              {chartData.map((e, i) => (
                <Cell
                  key={i}
                  fill={e.prediction === "FAKE" ? "#ef4444" : "#22c55e"}
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
            <XAxis dataKey="index" tick={{ fontSize: 10 }} />
            <YAxis
              yAxisId="left"
              unit="%"
              domain={[0, 100]}
              tick={{ fontSize: 10 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Area
              yAxisId="left"
              type="monotone"
              dataKey="score"
              fill="#8884d8"
              stroke="#8884d8"
              fillOpacity={0.1}
            />
            <Line
              yAxisId="left"
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
    <ResponsiveContainer width="100%" height={300}>
      {renderChart()}
    </ResponsiveContainer>
  );
};
