// src/components/analysis/charts/audio/SpectrogramHeatmapChart.jsx

import React, { useMemo, useState } from "react";
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  ZAxis,
  Cell,
} from "recharts";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "../../../ui/Card";
import { BarChart3, Activity } from "lucide-react";

// ShadCN UI components - you would import your actual components here
// For example:
// import { Slider } from "@/components/ui/slider";
// import { Label } from "@/components/ui/label";

// --- Mock ShadCN components for demonstration ---
const Slider = ({ value, onValueChange, ...props }) => (
  <input
    type="range"
    min="1"
    max="100"
    value={value}
    onChange={(e) => onValueChange([parseInt(e.target.value, 10)])}
    {...props}
  />
);
const Label = ({ children, ...props }) => <label {...props}>{children}</label>;
// --- End Mock Components ---

const CustomTooltip = ({ active, payload }) => {
  if (active && payload?.length) {
    const data = payload[0].payload;
    return (
      <div className="p-2 bg-gray-800 text-white rounded-md shadow-lg border border-gray-700 text-xs">
        <p>Time Bin: {data.time}</p>
        <p>Frequency Bin: {data.freq}</p>
        <p>Power (dB): {data.value.toFixed(2)}</p>
      </div>
    );
  }
  return null;
};

export const SpectrogramHeatmapChart = ({ data }) => {
  // State to control the data density (1% to 100%)
  const [granularity, setGranularity] = useState(20); // Default to 20%

  // Memoize the downsampled data transformation.
  // This recalculates only when the raw `data` or the `granularity` changes.
  const chartData = useMemo(() => {
    if (!data || data.length === 0 || !Array.isArray(data[0])) {
      return [];
    }

    const sampledData = [];
    const numRows = data.length;
    const numCols = data[0].length;

    // Calculate the step based on granularity to skip points.
    // A step of 1 means 100% of data, step of 10 means ~1% of data.
    const step = Math.max(1, Math.floor(100 / granularity));

    for (let i = 0; i < numRows; i += step) {
      for (let j = 0; j < numCols; j += step) {
        if (data[i] && data[i][j] !== undefined) {
          sampledData.push({
            time: j,
            freq: i,
            value: data[i][j],
          });
        }
      }
    }
    return sampledData;
  }, [data, granularity]);

  // `data` is expected to be a 2D array (matrix)
  if (!data || data.length === 0 || !Array.isArray(data[0])) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-primary-main" /> Interactive
            Spectrogram
          </CardTitle>
          <CardDescription>
            An interactive heatmap of the raw spectrogram data. Hover over
            points for details.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center p-8 text-light-muted-text dark:text-dark-muted-text">
            <Activity className="h-12 w-12 mx-auto mb-4" />
            <p className="text-lg font-semibold">No Raw Spectrogram Data</p>
            <p className="mt-2 text-sm">
              Raw spectrogram matrix data is not available for an interactive
              heatmap.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Simplified color scale
  const getColor = (value) => {
    const minDb = -80;
    const maxDb = 0;
    const normalized = Math.max(
      0,
      Math.min(1, (value - minDb) / (maxDb - minDb))
    );
    const hue = (1 - normalized) * 240; // Blue (cold) to Red (hot)
    return `hsl(${hue}, 80%, 50%)`;
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BarChart3 className="h-5 w-5 text-primary-main" /> Interactive
          Spectrogram
        </CardTitle>
        <CardDescription>
          An interactive heatmap of the raw spectrogram data. Use the slider to
          adjust detail.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid gap-4">
          <div className="flex items-center gap-4">
            <Label htmlFor="granularity-slider" className="w-24">
              Granularity: {granularity}%
            </Label>
            {/* 
              This is where you would place your ShadCN Slider.
              The `onValueChange` callback from ShadCN's slider provides an array, 
              so we take the first element.
            */}
            <Slider
              id="granularity-slider"
              min={1}
              max={100}
              step={1}
              value={[granularity]}
              onValueChange={(value) => setGranularity(value[0])}
            />
          </div>

          <p className="text-sm text-center text-light-muted-text dark:text-dark-muted-text">
            Rendering {chartData.length.toLocaleString()} points.
          </p>

          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <XAxis
                type="number"
                dataKey="time"
                name="Time"
                label={{
                  value: "Time Bins",
                  position: "insideBottom",
                  offset: -10,
                  fontSize: 12,
                  fill: "currentColor",
                }}
                tick={{ fontSize: 10, fill: "currentColor" }}
                domain={["dataMin", "dataMax"]}
              />
              <YAxis
                type="number"
                dataKey="freq"
                name="Frequency"
                label={{
                  value: "Frequency Bins",
                  angle: -90,
                  position: "insideLeft",
                  fontSize: 12,
                  fill: "currentColor",
                }}
                tick={{ fontSize: 10, fill: "currentColor" }}
                domain={["dataMin", "dataMax"]}
              />
              <ZAxis type="number" dataKey="value" range={[1, 100]} />
              <Tooltip
                content={<CustomTooltip />}
                cursor={{ strokeDasharray: "3 3" }}
              />
              <Scatter
                data={chartData}
                shape="square"
                isAnimationActive={false}
              >
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getColor(entry.value)} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
};
