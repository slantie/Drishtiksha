// src/pages/Analysis.jsx (Consolidated and Refined)

import React, { useEffect } from "react";
import { useParams, Link } from "react-router-dom";
import {
  ArrowLeft,
  RefreshCw,
  AlertTriangle,
  Video,
  Brain,
  SearchX,
  FileText,
  Clock,
  Shield,
} from "lucide-react";
import { useMediaItemQuery } from "../hooks/useMediaQuery.jsx";
import { Button } from "../components/ui/Button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardFooter,
} from "../components/ui/Card";
import { PageLoader } from "../components/ui/LoadingSpinner";
import { MediaPlayer } from "../components/media/MediaPlayer.jsx";
import { PageHeader } from "../components/layout/PageHeader.jsx";
import { AnalysisReport } from "../components/analysis/AnalysisReport.jsx";
import { EmptyState } from "../components/ui/EmptyState.jsx";
import { StatusBadge, Badge } from "../components/ui/Badge";
import { formatDate } from "../utils/formatters.js";

const AnalysisPage = () => {
  const { mediaId, analysisId } = useParams();
  const {
    data: media,
    isLoading,
    error,
    refetch,
    isRefetching,
  } = useMediaItemQuery(mediaId);

  useEffect(() => {
    document.title = "Analysis Report - Drishtiksha";
  }, []);

  // --- Loading State ---
  if (isLoading)
    return <PageLoader text="Loading Detailed Analysis Report..." />;

  // --- Error Loading Media Data ---
  if (error)
    return (
      <div className="text-center p-8 w-full max-w-full mx-auto space-y-4">
        <AlertTriangle className="w-16 h-16 text-red-500 mx-auto" />
        <h2 className="text-2xl font-bold">Error Loading Media</h2>
        <p className="text-light-muted-text dark:text-dark-muted-text">
          {error.message ||
            "An unexpected error occurred while fetching media details."}
        </p>
        <Button
          onClick={() => refetch()}
          variant="outline"
          className="mt-4"
          aria-label="Try again to load media"
        >
          <RefreshCw className="mr-2 h-4 w-4" /> Try Again
        </Button>
      </div>
    );

  // --- Media Not Found ---
  if (!media)
    return (
      <div className="text-center p-8 w-full max-w-full mx-auto space-y-4">
        <EmptyState
          icon={SearchX}
          title="Media Not Found"
          message="The requested media item could not be found or you do not have access."
          action={
            <Button asChild aria-label="Go Back">
              <Link to="/dashboard">
                <ArrowLeft className="mr-2 h-4 w-4" />Go Back
              </Link>
            </Button>
          }
        />
      </div>
    );

  // Find the specific analysis run
  const analysis = media.analysisRuns
    ?.flatMap((run) => run.analyses)
    .find((a) => a.id === analysisId);

  // Find the run that contains this analysis
  const analysisRun = media.analysisRuns?.find((run) =>
    run.analyses?.some((a) => a.id === analysisId)
  );

  // --- Specific Analysis Not Found ---
  if (!analysis) {
    return (
      <div className="text-center p-8 w-full max-w-full mx-auto space-y-4">
        <EmptyState
          icon={Brain}
          title="Analysis Not Found"
          message="The requested detailed analysis could not be found for this media item, or it has not been completed yet."
          action={
            <Button
              asChild
              variant="outline"
              aria-label="Go Back"
            >
              <Link to={`/results/${mediaId}`}>
                <ArrowLeft className="mr-2 h-4 w-4" /> Go Back
              </Link>
            </Button>
          }
        />
      </div>
    );
  }

  const resultPayload = analysis.resultPayload;
  const prediction = resultPayload?.prediction || analysis.prediction || "N/A";
  const confidence = resultPayload?.confidence || analysis.confidence || 0;
  const isReal = prediction === "REAL";

  return (
    <div className="space-y-6 w-full max-w-full mx-auto">
      <PageHeader
        title={`${analysis.modelName} - Detailed Report`}
        description={`A deep-dive forensic analysis for the file: ${media.filename}`}
        actions={
          <div className="flex flex-col sm:flex-row gap-2">
            <Button
              onClick={() => refetch()}
              isLoading={isRefetching}
              variant="outline"
              aria-label="Refresh analysis data"
            >
              <RefreshCw className="mr-2 h-4 w-4" /> Refresh Data
            </Button>
            <Button asChild variant="outline">
              <Link
                to={`/results/${mediaId}`}
                aria-label="Go Back"
              >
                <ArrowLeft className="mr-2 h-4 w-4" /> Go Back
              </Link>
            </Button>
          </div>
        }
      />

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 items-start animate-in fade-in slide-in-from-bottom-4 duration-500">
        {/* LEFT COLUMN (SIDEBAR) */}
        <div className="lg:col-span-1 lg:sticky lg:top-24 space-y-4">
          {/* Media Player Card */}
          <Card className="overflow-hidden transition-all duration-200 hover:shadow-lg">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Video className="h-4 w-4 text-primary-main" /> Original Media
              </CardTitle>
            </CardHeader>
            <CardContent className="p-2">
              <MediaPlayer media={media} />
            </CardContent>
          </Card>

          {/* Analysis Summary Card */}
          <Card
            className={`transition-all duration-200 hover:shadow-lg`}
          >
            <CardHeader>
              <CardTitle className="text-base w-full flex items-center gap-2">
                <div className="flex items-center w-full justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <Shield className={`h-5 w-5
                  ${
                      isReal
                        ? "text-green-600 dark:text-green-400"
                        : "text-red-600 dark:text-red-400"
                    }`} />
                Analysis Summary
                  </div>
                  <div>
                    <p
                    className={`text-lg ${
                      isReal
                        ? "text-green-600 dark:text-green-400"
                        : "text-red-600 dark:text-red-400"
                    }`}
                  >
                    {prediction}
                  </p>
                  </div>
                </div>

              </CardTitle>
            </CardHeader>
            <CardContent>
              {/* Prediction & Confidence */}
              

              {/* Analysis Details */}
              <div>
                <div className="flex items-center justify-between transition-colors duration-150 hover:bg-light-card-hover dark:hover:bg-dark-card-hover p-2 rounded -mx-2">
                  <span className="text-sm text-light-muted-text dark:text-dark-muted-text">
                    Model
                  </span>
                  <span className="text-sm font-medium">
                    {analysis.modelName}
                  </span>
                </div>
                <div className="flex items-center justify-between transition-colors duration-150 hover:bg-light-card-hover dark:hover:bg-dark-card-hover p-2 rounded -mx-2">
                  <span className="text-sm text-light-muted-text dark:text-dark-muted-text">
                    Confidence
                  </span>
                  <span className="text-sm font-medium">
                     {(confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex items-center justify-between transition-colors duration-150 hover:bg-light-card-hover dark:hover:bg-dark-card-hover p-2 rounded -mx-2">
                  <span className="text-sm text-light-muted-text dark:text-dark-muted-text">
                    Status
                  </span>
                  <StatusBadge status={analysis.status} />
                </div>
                {analysisRun && (
                  <div className="flex items-center justify-between transition-colors duration-150 hover:bg-light-card-hover dark:hover:bg-dark-card-hover p-2 rounded -mx-2">
                    <span className="text-sm text-light-muted-text dark:text-dark-muted-text">
                      Run Number
                    </span>
                    <Badge variant="secondary" size="sm">
                      #{analysisRun.runNumber}
                    </Badge>
                  </div>
                )}
                <div className="flex items-center justify-between transition-colors duration-150 hover:bg-light-card-hover dark:hover:bg-dark-card-hover p-2 rounded -mx-2">
                  <span className="text-sm text-light-muted-text dark:text-dark-muted-text flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    Analyzed
                  </span>
                  <span className="text-sm font-medium">
                    {formatDate(analysis.createdAt)}
                  </span>
                </div>
              </div>
            </CardContent>
            <CardFooter className="px-3 pb-3 pt-0 border-0">
              <Button
                asChild
                variant="outline"
                className="w-full transition-all duration-200 hover:scale-[1.02] active:scale-[0.98]"
                size="sm"
              >
                <Link to={`/results/${mediaId}`}>
                  <FileText className="h-4 w-4 mr-2" />
                  View All Analyses
                </Link>
              </Button>
            </CardFooter>
          </Card>
        </div>

        {/* RIGHT COLUMN (MAIN CONTENT) */}
        <div className="lg:col-span-3 animate-in fade-in slide-in-from-right-4 duration-500 delay-150">
          <AnalysisReport result={analysis.resultPayload} />
        </div>
      </div>
    </div>
  );
};

export default AnalysisPage;