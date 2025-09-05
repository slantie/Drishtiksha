// src/components/media/AnalysisComplete.jsx

import React from "react";
import { Link } from "react-router-dom";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
} from "../ui/Card.jsx";
import { Button } from "../ui/Button.jsx";
// REMOVED: import { useServerStatusQuery } from "../../hooks/useMonitoringQuery.js"; // Not used here
import {
  Eye,
  ShieldCheck,
  ShieldAlert,
  Clock,
  AlertTriangle,
  Bot,
} from "lucide-react";
import { formatProcessingTime } from "../../utils/formatters.js";
import { EmptyState } from "../ui/EmptyState.jsx";
import { Alert, AlertDescription, AlertTitle } from "../ui/Alert.jsx";
import { AnalysisResultSummaryCard } from "../analysis/charts/AnalysisResultSummaryCard.jsx"; // Ensure correct import path

export const AnalysisComplete = ({ run }) => {
  // Ensure `run` is valid before attempting to access its properties
  if (!run || !Array.isArray(run.analyses)) {
    return (
      <EmptyState
        icon={Bot}
        title="No Results"
        message="No analysis results were found for this run or run data is incomplete."
      />
    );
  }

  const analyses = [...run.analyses].sort(
    (a, b) => new Date(b.createdAt) - new Date(a.createdAt)
  );
  const successfulCount = analyses.filter(
    (a) => a.status === "COMPLETED"
  ).length;
  const failedCount = analyses.length - successfulCount;

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-3xl font-bold">
          Analysis Run #{run.runNumber} Complete
        </h2>
        <p className="text-light-muted-text dark:text-dark-muted-text mt-1">
          Found {successfulCount} successful result(s) and {failedCount} failed
          attempt(s).
        </p>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {analyses.map((analysis) => (
          <AnalysisResultSummaryCard
            key={analysis.id}
            analysis={analysis}
            mediaId={run.mediaId}
          />
        ))}
      </div>
    </div>
  );
};
