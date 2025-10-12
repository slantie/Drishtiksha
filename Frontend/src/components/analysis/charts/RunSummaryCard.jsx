// src/components/analysis/charts/RunSummaryCard.jsx
// Summary card showing key metrics for an analysis run

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "../../ui/Card";
import { Badge } from "../../ui/Badge";
import { CheckCircle, XCircle, AlertTriangle, TrendingUp } from "lucide-react";

export const RunSummaryCard = ({
  completedCount,
  failedCount,
  realCount,
  fakeCount,
  avgConfidence,
}) => {
  const totalModels = completedCount + failedCount;
  const successRate =
    totalModels > 0 ? (completedCount / totalModels) * 100 : 0;
  const overallVerdict = fakeCount > realCount ? "DEEPFAKE" : "AUTHENTIC";
  const consensusPercentage =
    completedCount > 0
      ? (Math.max(realCount, fakeCount) / completedCount) * 100
      : 0;

  return (
    <Card className="h-full flex flex-col">
      <CardHeader>
        <CardTitle className="text-2xl">Analysis Summary</CardTitle>
      </CardHeader>
      <CardContent className="flex-1">
        <div className="flex flex-col justify-between py-4 h-full">
          <div className="flex items-center justify-evenly w-full">
            {/* Overall Verdict */}
            <div className="space-y-2">
              <div className="text-sm font-medium text-light-muted-text dark:text-dark-muted-text">
                Overall Verdict
              </div>
              <Badge
                variant={overallVerdict === "AUTHENTIC" ? "success" : "danger"}
                className="text-lg py-2 px-4"
              >
                {overallVerdict}
              </Badge>
              <div className="text-xs text-light-muted-text dark:text-dark-muted-text">
                {consensusPercentage.toFixed(0)}% model agreement
              </div>
            </div>

            {/* Average Confidence */}
            <div className="space-y-2">
              <div className="text-sm font-medium text-light-muted-text dark:text-dark-muted-text flex items-center gap-2">
                <TrendingUp className="h-4 w-4" />
                Avg. Confidence
              </div>
              <div className="text-3xl font-bold">
                {(avgConfidence * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-light-muted-text dark:text-dark-muted-text">
                Across {completedCount} model{completedCount !== 1 ? "s" : ""}
              </div>
            </div>

            {/* Success Rate */}
            <div className="space-y-2">
              <div className="text-sm font-medium text-light-muted-text dark:text-dark-muted-text flex items-center gap-2">
                <CheckCircle className="h-4 w-4" />
                Success Rate
              </div>
              <div className="text-3xl font-bold text-green-600 dark:text-green-400">
                {successRate.toFixed(0)}%
              </div>
              <div className="text-xs text-light-muted-text dark:text-dark-muted-text">
                {completedCount}/{totalModels} models completed
              </div>
            </div>

            {/* Detection Results */}
            <div className="space-y-2">
              <div className="text-sm font-medium text-light-muted-text dark:text-dark-muted-text">
                Detection Results
              </div>
              <div className="flex gap-3">
                <div className="flex items-center gap-1">
                  <CheckCircle className="h-4 w-4 text-green-600 dark:text-green-400" />
                  <span className="text-sm font-semibold">
                    {realCount} REAL
                  </span>
                </div>
                <div className="flex items-center gap-1">
                  <AlertTriangle className="h-4 w-4 text-red-600 dark:text-red-400" />
                  <span className="text-sm font-semibold">
                    {fakeCount} FAKE
                  </span>
                </div>
              </div>
              {failedCount > 0 && (
                <div className="flex items-center gap-1 text-amber-600 dark:text-amber-400">
                  <XCircle className="h-4 w-4" />
                  <span className="text-xs">{failedCount} failed</span>
                </div>
              )}
            </div>
          </div>

          {/* Progress Bar */}
          <div className="px-4">
            <div className="flex justify-between text-xs text-light-muted-text dark:text-dark-muted-text mb-2">
              <span>Model Results</span>
              <span>
                {completedCount} of {totalModels}
              </span>
            </div>
            <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden flex">
              {realCount > 0 && (
                <div
                  className="bg-green-500 flex items-center justify-center text-[10px] text-white font-semibold"
                  style={{ width: `${(realCount / completedCount) * 100}%` }}
                  title={`${realCount} REAL`}
                >
                  {realCount}
                </div>
              )}
              {fakeCount > 0 && (
                <div
                  className="bg-red-500 flex items-center justify-center text-[10px] text-white font-semibold"
                  style={{ width: `${(fakeCount / completedCount) * 100}%` }}
                  title={`${fakeCount} FAKE`}
                >
                  {fakeCount}
                </div>
              )}
            </div>
            <div className="flex justify-between text-xs text-light-muted-text dark:text-dark-muted-text mt-1">
              <span className="text-green-600 dark:text-green-400">
                {realCount} Authentic
              </span>
              <span className="text-red-600 dark:text-red-400">
                {fakeCount} Deepfake
              </span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
