// src/pages/Analysis.jsx (Consolidated and Refined)

import React from "react";
import { useParams, Link } from "react-router-dom";
import {
  ArrowLeft,
  RefreshCw,
  AlertTriangle,
  Video,
  Brain,
  SearchX,
} from "lucide-react"; // Added SearchX for not found
import { useMediaItemQuery } from "../hooks/useMediaQuery.jsx";
import { Button } from "../components/ui/Button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "../components/ui/Card";
import { PageLoader } from "../components/ui/LoadingSpinner";
import { MediaPlayer } from "../components/media/MediaPlayer.jsx";
import { PageHeader } from "../components/layout/PageHeader.jsx";
// Removed showToast as it's not directly used for display logic here, but through hooks.
import { AnalysisReport } from "../components/analysis/AnalysisReport.jsx";
import { EmptyState } from "../components/ui/EmptyState.jsx"; // Import EmptyState for consistent error messages

const AnalysisPage = () => {
  // Renamed component for clarity
  const { mediaId, analysisId } = useParams();
  const {
    data: media,
    isLoading,
    error,
    refetch,
    isRefetching,
  } = useMediaItemQuery(mediaId);

  // --- Loading State ---
  if (isLoading)
    return <PageLoader text="Loading Detailed Analysis Report..." />;

  // --- Error Loading Media Data ---
  if (error)
    return (
      <div className="text-center p-8 w-full max-w-full mx-auto space-y-4">
        {" "}
        {/* Consistent full width and spacing */}
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
    // If data is null after loading and no error
    return (
      <div className="text-center p-8 w-full max-w-full mx-auto space-y-4">
        {" "}
        {/* Consistent full width and spacing */}
        <EmptyState
          icon={SearchX} // Specific icon for not found
          title="Media Not Found"
          message="The requested media item could not be found or you do not have access."
          action={
            <Button asChild aria-label="Back to dashboard">
              <Link to="/dashboard">
                <ArrowLeft className="mr-2 h-4 w-4" /> Back to Dashboard
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

  // --- Specific Analysis Not Found ---
  if (!analysis) {
    return (
      <div className="text-center p-8 w-full max-w-full mx-auto space-y-4">
        {" "}
        {/* Consistent full width and spacing */}
        <EmptyState
          icon={Brain} // Brain icon for analysis context
          title="Analysis Not Found"
          message="The requested detailed analysis could not be found for this media item, or it has not been completed yet."
          action={
            <Button
              asChild
              variant="outline"
              aria-label="Back to analysis summary"
            >
              <Link to={`/results/${mediaId}`}>
                <ArrowLeft className="mr-2 h-4 w-4" /> Back to Summary
              </Link>
            </Button>
          }
        />
      </div>
    );
  }

  return (
    <div className="space-y-6 w-full max-w-full mx-auto">
      {" "}
      {/* Consistent vertical spacing, full width */}
      <PageHeader
        title={`${analysis.modelName} - Detailed Report`}
        description={`A deep-dive forensic analysis for the file: ${media.filename}`}
        actions={
          <div className="flex flex-col sm:flex-row gap-2">
            {" "}
            {/* Responsive button grouping */}
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
                aria-label="Back to analysis summary"
              >
                <ArrowLeft className="mr-2 h-4 w-4" /> Back to Summary
              </Link>
            </Button>
          </div>
        }
      />
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Video className="h-5 w-5 text-primary-main" /> Original Media
          </CardTitle>
        </CardHeader>
        <CardContent>
          <MediaPlayer media={media} />
        </CardContent>
      </Card>
      {/* The main analysis report component */}
      <AnalysisReport result={analysis.resultPayload} />
    </div>
  );
};

export default AnalysisPage; // Export with the new name
