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
import { Film as VideoIcon, Activity } from "lucide-react"; // Changed LineChartIcon to Film for video context

// Import dedicated video chart components
import { ConfidenceAreaChart } from "./ConfidenceAreaChart";
import { PredictionBarChart } from "./PredictionBarChart";
import { TrendlineAnalysisChart } from "./TrendLineAnalysisChart.jsx";
import { TemporalHeatmap } from "./TemporalHeatmap";
import { ConfidenceDistributionChart } from "./ConfidenceDistributionChart"; // New component for distribution

export const VideoReport = ({ result }) => {
  // `result` here is the analysis.resultPayload
  // Extract frame predictions and metrics from the resultPayload
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
              This model did not provide frame-level predictions, or the media
              could not be processed for detailed analysis.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Transform frame predictions to a format suitable for Recharts if needed
  // Assuming each frame in framePredictions has 'index', 'score', 'prediction'
  // If backend sends 'confidence' instead of 'score', adjust here:
  const chartData = framePredictions.map((frame) => ({
    index: frame.frame_number, // Or simply 'index' if available
    score: frame.confidence, // Use 'confidence' if backend sends it
    prediction: frame.prediction,
  }));

  const realFrames = chartData.filter((f) => f.prediction === "REAL").length;
  const fakeFrames = chartData.filter((f) => f.prediction === "FAKE").length;
  const avgFakeScore = metrics.final_average_score
    ? metrics.final_average_score * 100
    : (chartData.reduce((sum, f) => sum + f.score, 0) / chartData.length) *
        100 || 0;

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
            <div className="text-2xl font-bold">{chartData.length}</div>
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
