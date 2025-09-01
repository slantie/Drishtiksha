// src/components/analysis/charts/SpectrogramHeatmapChart.jsx

import React, { useMemo } from "react";
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  ZAxis,
} from "recharts";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "../../../ui/Card";
import { BarChart3 } from "lucide-react";

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
  if (!data || data.length === 0) return null;

  // Transform the matrix into a flat array of points suitable for a scatter chart.
  const chartData = useMemo(
    () =>
      data.flatMap((row, freqIndex) =>
        row.map((value, timeIndex) => ({
          time: timeIndex,
          freq: freqIndex,
          value: value,
        }))
      ),
    [data]
  );

  // This is a simplified color scale. A more advanced version could use a library.
  const getColor = (value) => {
    const minDb = -80; // A typical floor for spectrograms
    const maxDb = 0;
    const normalized = Math.max(0, (value - minDb) / (maxDb - minDb));
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
          An interactive heatmap of the raw spectrogram data. Hover over points
          for details.
        </CardDescription>
      </CardHeader>
      <CardContent>
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
              }}
              tick={{ fontSize: 10 }}
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
              }}
              tick={{ fontSize: 10 }}
              domain={["dataMin", "dataMax"]}
            />
            <ZAxis type="number" dataKey="value" range={[1, 100]} />
            <Tooltip
              content={<CustomTooltip />}
              cursor={{ strokeDasharray: "3 3" }}
            />
            <Scatter data={chartData} fill="#8884d8" shape="square">
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={getColor(entry.value)} />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};
