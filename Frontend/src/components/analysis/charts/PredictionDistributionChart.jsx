// src/components/analysis/charts/PredictionDistributionChart.jsx
// Pie chart showing distribution of REAL vs FAKE predictions

import React from "react";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from "recharts";

export const PredictionDistributionChart = ({ realCount, fakeCount }) => {
  const data = [
    { name: "Authentic (REAL)", value: realCount, color: "#10b981" },
    { name: "Deepfake (FAKE)", value: fakeCount, color: "#ef4444" },
  ];

  // Filter out zero values
  const filteredData = data.filter((item) => item.value > 0);

  if (filteredData.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-light-muted-text dark:text-dark-muted-text">
        No predictions available
      </div>
    );
  }

  const renderLabel = (entry) => {
    const percentage = ((entry.value / (realCount + fakeCount)) * 100).toFixed(
      0
    );
    return `${entry.name}: ${entry.value} (${percentage}%)`;
  };

  return (
    <ResponsiveContainer width="100%" height={300}>
      <PieChart>
        <Pie
          data={filteredData}
          cx="50%"
          cy="50%"
          labelLine={false}
        //   label={renderLabel}
          outerRadius={80}
          fill="#8884d8"
          dataKey="value"
        >
          {filteredData.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.color} />
          ))}
        </Pie>
        {/* <Tooltip
          contentStyle={{
            backgroundColor: "rgba(255, 255, 255, 0.95)",
            border: "1px solid #e2e8f0",
            borderRadius: "0.375rem",
          }}
        /> */}
        <Legend
          verticalAlign="bottom"
          height={36}
          formatter={(value, entry) => (
            <span style={{ color: entry.color }}>{value}</span>
          )}
        />
      </PieChart>
    </ResponsiveContainer>
  );
};
