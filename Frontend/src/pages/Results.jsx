// src/pages/Results.jsx

import React, { useState } from "react";
import { useParams, Link } from "react-router-dom";
import {
  ArrowLeft,
  Edit,
  Trash2,
  Brain,
  AlertTriangle,
  Download,
  RefreshCw,
  Layers,
  FileText, // Added icon for download report
} from "lucide-react";
import {
  useMediaItemQuery,
  useDeleteMediaMutation,
} from "../hooks/useMediaQuery.jsx";
import { Button } from "../components/ui/Button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../components/ui/Card";
import { PageLoader } from "../components/ui/LoadingSpinner";
import { MediaPlayer } from "../components/media/MediaPlayer.jsx";
import { AnalysisInProgress } from "../components/media/AnalysisInProgress.jsx";
import { EditMediaModal } from "../components/media/EditMediaModal.jsx";
import { DeleteMediaModal } from "../components/media/DeleteMediaModal.jsx";
import { RerunAnalysisModal } from "../components/analysis/RerunAnalysisModal.jsx";
import { PageHeader } from "../components/layout/PageHeader.jsx";
import { formatDate } from "../utils/formatters.js";
import { EmptyState } from "../components/ui/EmptyState.jsx";
import { AnalysisResultSummaryCard } from "../components/analysis/charts/AnalysisResultSummaryCard.jsx";
import { DownloadService } from "../services/DownloadReport.js"; // Import the DownloadService
import { useAuth } from "../contexts/AuthContext.jsx"; // Import useAuth to get user for report generation

const AnalysisRunCard = ({ run, mediaId }) => (
  <Card>
    <CardHeader>
      <div className="flex justify-between items-center gap-4">
        {" "}
        {/* Added gap for responsiveness */}
        <CardTitle className="text-lg">
          Analysis Run #{run.runNumber}
        </CardTitle>{" "}
        {/* Consistent title size */}
        <span className="text-xs text-light-muted-text dark:text-dark-muted-text flex-shrink-0">
          {formatDate(run.createdAt)}
        </span>
      </div>
      <CardDescription>
        Status:{" "}
        <span className="capitalize font-semibold">
          {run.status.toLowerCase()}
        </span>
      </CardDescription>
    </CardHeader>
    <CardContent>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {" "}
        {/* Consistent gap */}
        {run.analyses?.length > 0 ? (
          run.analyses.map((analysis) => (
            <AnalysisResultSummaryCard
              key={analysis.id}
              analysis={analysis}
              mediaId={mediaId}
            />
          ))
        ) : (
          <div className="col-span-full text-center text-light-muted-text dark:text-dark-muted-text p-4">
            <Brain className="h-10 w-10 mx-auto mb-3" />
            <p>No completed analyses for this run yet.</p>
          </div>
        )}
      </div>
    </CardContent>
  </Card>
);

const Results = () => {
  const { mediaId } = useParams();
  const { user } = useAuth(); // Get user for report download
  const {
    data: media,
    isLoading,
    error,
    refetch,
    isRefetching, // Added isRefetching
  } = useMediaItemQuery(mediaId, {
    refetchInterval: (query) => {
      const currentMedia = query.state.data;
      const latestRunStatus =
        currentMedia?.analysisRuns?.[0]?.status || currentMedia?.status;
      return ["QUEUED", "PROCESSING"].includes(latestRunStatus) ? 5000 : false;
    },
  });
  const deleteMutation = useDeleteMediaMutation();
  const [modal, setModal] = useState({ type: null, data: null });

  // Handle various loading/error states for the main page
  if (isLoading) return <PageLoader text="Loading Media & Results..." />;
  if (error)
    return (
      <div className="text-center p-8 w-full max-w-full mx-auto">
        {" "}
        {/* Consistent full width */}
        <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
        <h2 className="text-2xl font-bold">Error Loading Media</h2>
        <p className="text-light-muted-text dark:text-dark-muted-text mt-2">
          {error.message ||
            "An unexpected error occurred while fetching media details."}
        </p>
        <Button onClick={() => refetch()} variant="outline" className="mt-4">
          <RefreshCw className="mr-2 h-4 w-4" /> Try Again
        </Button>
      </div>
    );
  if (!media)
    return (
      <div className="text-center p-8 w-full max-w-full mx-auto">
        {" "}
        {/* Consistent full width */}
        <AlertTriangle className="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <h2 className="text-2xl font-bold">Media Not Found</h2>
        <p className="text-light-muted-text dark:text-dark-muted-text mt-2">
          The requested media item could not be found or you do not have access.
        </p>
        <Button asChild className="mt-4">
          <Link to="/dashboard">
            <ArrowLeft className="mr-2 h-4 w-4" /> Back to Dashboard
          </Link>
        </Button>
      </div>
    );

  const latestRun = media.analysisRuns?.[0];
  const isProcessing = ["QUEUED", "PROCESSING", "PARTIALLY_ANALYZED"].includes(
    latestRun?.status || media.status
  ); // Include PARTIALLY_ANALYZED as still "in progress" from a user perspective

  return (
    <div className="space-y-6 w-full max-w-full mx-auto">
      {" "}
      {/* Consistent vertical spacing, full width */}
      <PageHeader
        title={media.filename}
        description={media.description || "Analysis results and history."}
        actions={
          <div className="flex flex-col sm:flex-row gap-2">
            {" "}
            {/* Responsive button grouping */}
            <Button
              onClick={() => refetch()}
              isLoading={isRefetching}
              variant="outline"
              aria-label="Refresh media data"
            >
              <RefreshCw className="mr-2 h-4 w-4" /> Refresh Data
            </Button>
            <Button asChild variant="outline">
              <Link to="/dashboard" aria-label="Back to dashboard">
                <ArrowLeft className="mr-2 h-4 w-4" /> Back to Dashboard
              </Link>
            </Button>
          </div>
        }
      />
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
        {" "}
        {/* Consistent gap */}
        <div className="lg:col-span-1 space-y-6 lg:sticky lg:top-24">
          {" "}
          {/* Consistent vertical spacing, sticky for actions */}
          <Card>
            <CardHeader>
              <CardTitle>Original Media</CardTitle>
            </CardHeader>
            <CardContent>
              <MediaPlayer media={media} />
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Actions</CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {" "}
              {/* Consistent gap, responsive columns */}
              <Button
                onClick={() => setModal({ type: "rerun", data: media })}
                aria-label={`Re-analyze ${media.filename}`}
              >
                <RefreshCw className="h-4 w-4 mr-2" /> Re-Analyze
              </Button>
              <Button
                onClick={() => setModal({ type: "edit", data: media })}
                variant="outline"
                aria-label={`Edit description for ${media.filename}`}
              >
                <Edit className="h-4 w-4 mr-2" /> Edit
              </Button>
              <Button
                onClick={() =>
                  DownloadService.downloadMedia(media.url, media.filename)
                } // Use DownloadService
                variant="outline"
                aria-label={`Download original file ${media.filename}`}
              >
                <Download className="h-4 w-4 mr-2" /> Download File
              </Button>
              {media.analysisRuns?.[0]?.analyses?.length >
                0 /* Only show if there's any analysis to report */ && (
                <Button
                  onClick={() =>
                    DownloadService.generateAndDownloadPDF(media, user)
                  } // Use DownloadService for PDF
                  variant="outline"
                  aria-label={`Download PDF report for ${media.filename}`}
                >
                  <FileText className="h-4 w-4 mr-2" /> Download Report
                </Button>
              )}
              <Button
                onClick={() => setModal({ type: "delete", data: media })}
                variant="destructive"
                aria-label={`Delete ${media.filename}`}
              >
                <Trash2 className="h-4 w-4 mr-2" /> Delete
              </Button>
            </CardContent>
          </Card>
        </div>
        <div className="lg:col-span-2 space-y-6">
          {" "}
          {/* Consistent vertical spacing */}
          {isProcessing ? (
            <AnalysisInProgress media={media} />
          ) : media.analysisRuns && media.analysisRuns.length > 0 ? (
            <>
              <div className="flex items-center gap-2 text-light-muted-text dark:text-dark-muted-text">
                <Layers className="h-5 w-5" /> {/* Consistent icon size */}
                <h3 className="text-lg font-semibold">
                  Analysis History ({media.analysisRuns.length} Runs)
                </h3>
              </div>
              {media.analysisRuns.map((run) => (
                <AnalysisRunCard key={run.id} run={run} mediaId={media.id} />
              ))}
            </>
          ) : (
            <EmptyState
              icon={Brain}
              title="No Analysis Found"
              message="This media has not yet been analyzed or all analysis attempts have failed."
              action={
                <Button
                  onClick={() => setModal({ type: "rerun", data: media })}
                  aria-label={`Start first analysis for ${media.filename}`}
                >
                  <RefreshCw className="mr-2 h-4 w-4" /> Start First Analysis
                </Button>
              }
            />
          )}
        </div>
      </div>
      {/* --- MODALS --- */}
      <EditMediaModal
        isOpen={modal.type === "edit"}
        onClose={() => setModal({ type: null })}
        media={modal.data}
      />
      <DeleteMediaModal
        isOpen={modal.type === "delete"}
        onClose={() => setModal({ type: null })}
        media={modal.data}
      />
      <RerunAnalysisModal
        isOpen={modal.type === "rerun"}
        onClose={() => setModal({ type: null })}
        media={modal.data}
        onAnalysisStart={refetch}
      />
    </div>
  );
};

export default Results;
