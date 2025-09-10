// src/components/analysis/charts/video/VideoReport.jsx

import React from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "../../../ui/Card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "../../../ui/Tabs";
import { Film as VideoIcon, Activity } from "lucide-react";

import { ConfidenceAreaChart } from "./ConfidenceAreaChart";
import { PredictionBarChart } from "./PredictionBarChart";
import { TrendlineAnalysisChart } from "./TrendLineAnalysisChart.jsx";
import { TemporalHeatmap } from "./TemporalHeatmap";
import { ConfidenceDistributionChart } from "./ConfidenceDistributionChart";

export const VideoReport = ({ result }) => {
  const framePredictions = result.frame_predictions || [];
  const metrics = result.metrics || {};

  if (framePredictions.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <VideoIcon className="h-5 w-5 text-primary-main" /> Video Analysis
            Details
          </CardTitle>
          <CardDescription>
            Frame-by-frame insights into potential deepfake indicators.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center p-8 text-light-muted-text dark:text-dark-muted-text">
            <Activity className="h-12 w-12 mx-auto mb-4" />
            <p className="text-lg font-semibold">No Frame-by-Frame Data</p>
            <p className="mt-2 text-sm">
              This model did not provide frame-level predictions.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // CORRECTED: Use the 'index' and 'score' keys directly from the API response.
  const chartData = framePredictions.map((frame) => ({
    index: frame.index,
    score: frame.score,
    prediction: frame.prediction,
  }));

  const realFrames = chartData.filter((f) => f.prediction === "REAL").length;
  const fakeFrames = chartData.filter((f) => f.prediction === "FAKE").length;

  // IMPROVED: More robustly find the average score from the metrics object.
  const averageScoreKey = Object.keys(metrics).find(
    (key) =>
      key.includes("average") &&
      (key.includes("score") || key.includes("suspicion"))
  );
  const backendAvgScore = averageScoreKey ? metrics[averageScoreKey] : null;

  // Use the backend-provided average if available, otherwise calculate it as a fallback.
  const avgFakeScore =
    backendAvgScore !== null
      ? backendAvgScore * 100
      : (chartData.reduce((sum, f) => sum + f.score, 0) /
          (chartData.length || 1)) *
        100;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <VideoIcon className="h-5 w-5 text-primary-main" /> Video Analysis
          Details
        </CardTitle>
        <CardDescription>
          Frame-by-frame insights into potential deepfake indicators.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4 bg-light-muted-background dark:bg-dark-secondary rounded-lg mb-6 text-center">
          <div>
            <div className="text-2xl font-bold text-green-600">
              {realFrames}
            </div>
            <div className="text-xs">Authentic Frames</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-red-600">{fakeFrames}</div>
            <div className="text-xs">Deepfake Frames</div>
          </div>
          <div>
            <div className="text-2xl font-bold">
              {result.frames_analyzed || chartData.length}
            </div>
            <div className="text-xs">Total Analyzed</div>
          </div>
          <div>
            <div
              className={`text-2xl font-bold ${
                avgFakeScore > 50 ? "text-red-500" : "text-green-500"
              }`}
            >
              {avgFakeScore.toFixed(1)}%
            </div>
            <div className="text-xs">Avg. Fake Score</div>
          </div>
        </div>

        <Tabs defaultValue="area-plot">
          <TabsList className="grid w-full grid-cols-2 sm:grid-cols-3 md:grid-cols-5">
            <TabsTrigger value="area-plot">Area Plot</TabsTrigger>
            <TabsTrigger value="bar-chart">Bar Chart</TabsTrigger>
            <TabsTrigger value="trendline">Trend Line</TabsTrigger>
            <TabsTrigger value="heatmap">Heatmap</TabsTrigger>
            <TabsTrigger value="distribution">Distribution</TabsTrigger>
          </TabsList>
          <TabsContent value="area-plot">
            <ConfidenceAreaChart frames={chartData} />
          </TabsContent>
          <TabsContent value="bar-chart">
            <PredictionBarChart frames={chartData} />
          </TabsContent>
          <TabsContent value="trendline">
            <TrendlineAnalysisChart frames={chartData} />
          </TabsContent>
          <TabsContent value="heatmap">
            <TemporalHeatmap frames={chartData} />
          </TabsContent>
          <TabsContent value="distribution">
            <ConfidenceDistributionChart frames={chartData} />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};
