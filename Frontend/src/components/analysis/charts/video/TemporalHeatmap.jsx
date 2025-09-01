// src/components/analysis/charts/video/TemporalHeatmap.jsx

import React from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "../../../ui/Card";
import { Thermometer } from "lucide-react";

export const TemporalHeatmap = ({ frames }) => {
  if (!frames || frames.length === 0) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Thermometer className="h-5 w-5 text-primary-main" /> Temporal Heatmap
        </CardTitle>
        <CardDescription>
          An at-a-glance view of suspicion levels across the media's timeline.
          Darker red indicates a higher "fake" score.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="w-full h-16 flex rounded-lg overflow-hidden border dark:border-dark-secondary">
          {frames.map((frame) => (
            <div
              key={frame.index}
              className="flex-1 group relative"
              title={`Frame ${frame.index}: ${(frame.score * 100).toFixed(
                1
              )}% Fake Score`}
              style={{
                backgroundColor:
                  frame.prediction === "REAL" ? "#22c55e" : "#ef4444",
                opacity: 0.15 + frame.score * 0.85,
              }}
            >
              <div className="absolute bottom-full mb-2 w-max p-2 text-xs bg-gray-800 text-white rounded-md opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none -translate-x-1/2 left-1/2 z-10">
                Frame {frame.index}: {(frame.score * 100).toFixed(1)}%
              </div>
            </div>
          ))}
        </div>
        <div className="flex justify-between text-xs text-light-muted-text dark:text-dark-muted-text mt-2 px-1">
          <span>Start of Timeline</span>
          <span>End of Timeline</span>
        </div>
      </CardContent>
    </Card>
  );
};
