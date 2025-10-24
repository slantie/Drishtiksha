// src/pages/Results.jsx

import React, { useState, useEffect } from "react";
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
  FileText,
  ChevronDown,
  Video,
  Eye,
} from "lucide-react";
import {
  useMediaItemQuery,
  useDeleteMediaMutation,
} from "../hooks/useMediaQuery.jsx";
import { Button } from "../components/ui/Button";
import { StatusBadge, MediaTypeBadge, Badge } from "../components/ui/Badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  CardFooter,
} from "../components/ui/Card";
import { mediaTypeConfigs } from "../constants/badgeConstants";
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
import { MarkdownPDFService } from "../services/pdf/MarkdownPDFService.js";
import { MediaDownloadService } from "../services/MediaDownloadService.js";
import { useAuth } from "../hooks/useAuth.js";
import { showToast } from "../utils/toast.jsx";

const AnalysisRunCard = ({ run, mediaId, isExpanded, onToggle }) => {
  const [isDownloading, setIsDownloading] = useState(false);

  const handleDownloadReport = async (e) => {
    e.stopPropagation(); // Prevent card toggle
    setIsDownloading(true);
    try {
      await MarkdownPDFService.downloadPDF(run.id);
      showToast.success(`Downloaded report for Run #${run.runNumber}`);
    } catch (error) {
      console.error("PDF download error:", error);
      showToast.error(`Failed to download report: ${error.message}`);
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <Card className="overflow-hidden transition-all duration-200 hover:shadow-lg">
      <button
        onClick={onToggle}
        className="w-full text-left hover:bg-light-card-hover dark:hover:bg-dark-card-hover transition-colors duration-150"
      >
        <CardHeader className="cursor-pointer">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <div className="flex items-center gap-3">
              <CardTitle className="text-lg flex items-center gap-2">
                Run #{run.runNumber}
                <ChevronDown
                  className={`h-5 w-5 text-light-muted-text dark:text-dark-muted-text transition-transform duration-300 ${
                    isExpanded ? "rotate-0" : "-rotate-90"
                  }`}
                />
              </CardTitle>
              <StatusBadge status={run.status} />
              {run.analyses?.length > 0 && (
                <Badge
                  variant="secondary"
                  size="sm"
                  className="transition-opacity duration-200"
                >
                  {run.analyses.length}{" "}
                  {run.analyses.length === 1 ? "Analysis" : "Analyses"}
                </Badge>
              )}
            </div>
            <div className="flex items-center gap-2">
              {run.analyses?.length > 0 && (
                <>
                  <Button
                    asChild
                    variant="outline"
                    size="sm"
                    className="transition-all duration-200 hover:scale-[1.02] active:scale-[0.98]"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <Link to={`/results/${mediaId}/runs/${run.id}`}>
                      <Layers className="h-4 w-4 mr-2" /> View Details
                    </Link>
                  </Button>
                  <Button
                    onClick={handleDownloadReport}
                    variant="outline"
                    size="sm"
                    disabled={isDownloading}
                    aria-label={`Download report for Run #${run.runNumber}`}
                    className="transition-all duration-200 hover:scale-[1.02] active:scale-[0.98]"
                  >
                    {isDownloading ? (
                      <>
                        <RefreshCw className="h-4 w-4 mr-2 animate-spin" />{" "}
                        Generating...
                      </>
                    ) : (
                      <>
                        <Download className="h-4 w-4 mr-2" /> PDF
                      </>
                    )}
                  </Button>
                </>
              )}
              <span className="text-xs text-light-muted-text dark:text-dark-muted-text">
                {formatDate(run.createdAt)}
              </span>
            </div>
          </div>
        </CardHeader>
      </button>
      <div
        className={`transition-all duration-300 ease-in-out ${
          isExpanded
            ? "max-h-[5000px] opacity-100"
            : "max-h-0 opacity-0 overflow-hidden"
        }`}
      >
        <CardContent>
          <div className="space-y-3">
            {run.analyses?.length > 0 ? (
              run.analyses.map((analysis, index) => (
                <div
                  key={analysis.id}
                  className="animate-in fade-in slide-in-from-top-2 duration-300"
                  style={{ animationDelay: `${index * 50}ms` }}
                >
                  <AnalysisResultSummaryCard
                    analysis={analysis}
                    mediaId={mediaId}
                  />
                </div>
              ))
            ) : (
              <div className="text-center text-light-muted-text dark:text-dark-muted-text p-8 animate-in fade-in duration-300">
                <Brain className="h-10 w-10 mx-auto mb-3" />
                <p className="text-sm">
                  No completed analyses for this run yet.
                </p>
              </div>
            )}
          </div>
        </CardContent>
      </div>
    </Card>
  );
};

const Results = () => {
  const { mediaId } = useParams();
  const { user } = useAuth();
  const {
    data: media,
    isLoading,
    error,
    refetch,
    isRefetching,
    dataUpdatedAt,
  } = useMediaItemQuery(mediaId, {
    refetchInterval: (query) => {
      const currentMedia = query.state.data;
      const latestRunStatus =
        currentMedia?.analysisRuns?.[0]?.status || currentMedia?.status;

      if (!["QUEUED", "PROCESSING"].includes(latestRunStatus)) {
        return false;
      }

      const now = Date.now();
      const lastUpdate = query.state.dataUpdatedAt || now;
      const timeSinceUpdate = now - lastUpdate;

      if (timeSinceUpdate < 30000) return 2000;
      if (timeSinceUpdate < 150000) return 5000;
      return 10000;
    },
    refetchOnWindowFocus: true,
    staleTime: 2000,
  });
  const deleteMutation = useDeleteMediaMutation();
  const [modal, setModal] = useState({ type: null, data: null });
  const [expandedRuns, setExpandedRuns] = useState(new Set([0])); // Expand first run by default
  const [pdfGenerating, setPdfGenerating] = useState(false);

  const toggleRun = (index) => {
    setExpandedRuns((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(index)) {
        newSet.delete(index);
      } else {
        newSet.add(index);
      }
      return newSet;
    });
  };

  useEffect(() => {
    if (media) {
      document.title = `${media.filename} - Results - Drishtiksha`;
    } else {
      document.title = "Results - Drishtiksha";
    }
  }, [media]);

  if (isLoading) return <PageLoader text="Loading Media & Results..." />;
  if (error)
    return (
      <div className="text-center p-8 w-full max-w-full mx-auto">
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
        <AlertTriangle className="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <h2 className="text-2xl font-bold">Media Not Found</h2>
        <p className="text-light-muted-text dark:text-dark-muted-text mt-2">
          The requested media item could not be found or you do not have access.
        </p>
        <Button asChild className="mt-4">
          <Link to="/dashboard">
            <ArrowLeft className="mr-2 h-4 w-4" /> Back
          </Link>
        </Button>
      </div>
    );

  const latestRun = media.analysisRuns?.[0];
  const isProcessing = ["QUEUED", "PROCESSING", "PARTIALLY_ANALYZED"].includes(
    latestRun?.status || media.status
  );

  const formatFileSize = (bytes) => {
    if (!bytes) return "N/A";
    const sizes = ["Bytes", "KB", "MB", "GB"];
    if (bytes === 0) return "0 Bytes";
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round((bytes / Math.pow(1024, i)) * 100) / 100 + " " + sizes[i];
  };

  return (
    <div className="space-y-6 w-full max-w-full mx-auto">
      <PageHeader
        title={media.filename}
        description={media.description || "Analysis results and history."}
        actions={
          <div className="flex flex-col sm:flex-row gap-2">
            <Button
              onClick={() => refetch()}
              isLoading={isRefetching}
              variant="outline"
              aria-label="Refresh media data"
            >
              <RefreshCw className="mr-2 h-4 w-4" /> Refresh Data
            </Button>
            <Button asChild variant="outline">
              <Link to="/dashboard" aria-label="Go Back">
                <ArrowLeft className="mr-2 h-4 w-4" /> Go Back
              </Link>
            </Button>
          </div>
        }
      />

      {/* Two-Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 items-start animate-in fade-in slide-in-from-bottom-4 duration-500">
        {/* LEFT COLUMN (SIDEBAR) */}
        <div className="lg:col-span-1 lg:sticky lg:top-24 space-y-4">
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
          <Card className="transition-all duration-200 hover:shadow-lg">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">Media Details</CardTitle>
                <StatusBadge status={latestRun?.status || media.status} />
              </div>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="flex items-center justify-between transition-colors duration-150 hover:bg-light-card-hover dark:hover:bg-dark-card-hover p-2 rounded -mx-2">
                <span className="text-sm text-light-muted-text dark:text-dark-muted-text">
                  File Size
                </span>
                <span className="text-sm font-medium">
                  {formatFileSize(media.size)}
                </span>
              </div>
              <div className="flex items-center justify-between transition-colors duration-150 hover:bg-light-card-hover dark:hover:bg-dark-card-hover p-2 rounded -mx-2">
                <span className="text-sm text-light-muted-text dark:text-dark-muted-text">
                  Uploaded
                </span>
                <span className="text-sm font-medium">
                  {formatDate(media.createdAt)}
                </span>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* RIGHT COLUMN (MAIN CONTENT) */}
        <Card className="lg:col-span-3 space-y-4 p-4">
          {isProcessing ? (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
              <AnalysisInProgress media={media} />
            </div>
          ) : media.analysisRuns && media.analysisRuns.length > 0 ? (
            <>
              <CardTitle className="flex items-center gap-2">
                <div className="flex items-center gap-2">
                  <Layers className="h-5 w-5" />
                  <h3 className="text-lg font-semibold">
                    Analysis History ({media.analysisRuns.length} Runs)
                  </h3>
                </div>
                <div className="flex items-center justify-between gap-2 ml-auto">
                  <Button
                    onClick={() => setModal({ type: "rerun", data: media })}
                    className="w-full transition-all duration-200 hover:scale-[1.02] active:scale-[0.98]"
                    size="sm"
                    aria-label={`Re-analyze ${media.filename}`}
                  >
                    <RefreshCw className="h-4 w-4 mr-2" /> Re-Analyze
                  </Button>
                  <Button
                    onClick={() => setModal({ type: "edit", data: media })}
                    variant="outline"
                    className="w-full transition-all duration-200 hover:scale-[1.02] active:scale-[0.98]"
                    size="sm"
                    aria-label={`Edit description for ${media.filename}`}
                  >
                    <Edit className="h-4 w-4 mr-2" /> Edit
                  </Button>
                  <Button
                    onClick={() =>
                      MediaDownloadService.downloadMedia(
                        media.url,
                        media.filename
                      )
                    }
                    variant="outline"
                    className="w-full transition-all duration-200 hover:scale-[1.02] active:scale-[0.98]"
                    size="sm"
                    aria-label={`Download original file ${media.filename}`}
                  >
                    <Download className="h-4 w-4 mr-2" /> Download
                  </Button>
                  {media.analysisRuns?.[0]?.analyses?.length > 0 && (
                    <>
                      <Button
                        onClick={async () => {
                          setPdfGenerating(true);
                          try {
                            // Get the latest/first analysis run
                            const latestRun = media.analysisRuns?.[0];
                            if (!latestRun) {
                              throw new Error("No analysis run available");
                            }
                            await MarkdownPDFService.previewPDF(latestRun.id);
                          } catch (error) {
                            console.error("PDF preview error:", error);
                          } finally {
                            setPdfGenerating(false);
                          }
                        }}
                        variant="outline"
                        className="w-full transition-all duration-200 hover:scale-[1.02] active:scale-[0.98]"
                        size="sm"
                        disabled={pdfGenerating}
                        aria-label={`Preview PDF report for ${media.filename}`}
                      >
                        {pdfGenerating ? (
                          <>
                            <RefreshCw className="h-4 w-4 mr-2 animate-spin" />{" "}
                            Loading...
                          </>
                        ) : (
                          <>
                            <Eye className="h-4 w-4 mr-2" /> Preview
                          </>
                        )}
                      </Button>
                      <Button
                        onClick={async () => {
                          setPdfGenerating(true);
                          try {
                            // Get the latest/first analysis run
                            const latestRun = media.analysisRuns?.[0];
                            if (!latestRun) {
                              throw new Error("No analysis run available");
                            }
                            await MarkdownPDFService.downloadPDF(latestRun.id);
                          } catch (error) {
                            console.error("PDF download error:", error);
                          } finally {
                            setPdfGenerating(false);
                          }
                        }}
                        variant="outline"
                        className="w-full transition-all duration-200 hover:scale-[1.02] active:scale-[0.98]"
                        size="sm"
                        disabled={pdfGenerating || !media.analysisRuns?.[0]}
                        aria-label={`Download PDF report for ${media.filename}`}
                      >
                        {pdfGenerating ? (
                          <>
                            <RefreshCw className="h-4 w-4 mr-2 animate-spin" />{" "}
                            Generating...
                          </>
                        ) : (
                          <>
                            <FileText className="h-4 w-4 mr-2" /> Latest Report
                          </>
                        )}
                      </Button>
                    </>
                  )}
                  <Button
                    onClick={() => setModal({ type: "delete", data: media })}
                    variant="destructive"
                    className="w-full transition-all duration-200 hover:scale-[1.02] active:scale-[0.98]"
                    size="sm"
                    aria-label={`Delete ${media.filename}`}
                  >
                    <Trash2 className="h-4 w-4 mr-2" /> Delete
                  </Button>
                </div>
              </CardTitle>
              {media.analysisRuns.map((run, index) => (
                <div
                  key={run.id}
                  className="animate-in fade-in slide-in-from-bottom-2 duration-500"
                  style={{ animationDelay: `${index * 100}ms` }}
                >
                  <AnalysisRunCard
                    run={run}
                    mediaId={media.id}
                    isExpanded={expandedRuns.has(index)}
                    onToggle={() => toggleRun(index)}
                  />
                </div>
              ))}
            </>
          ) : (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
              <EmptyState
                icon={Brain}
                title="No Analysis Found"
                message="This media has not yet been analyzed or all analysis attempts have failed."
                action={
                  <Button
                    onClick={() => setModal({ type: "rerun", data: media })}
                    aria-label={`Start first analysis for ${media.filename}`}
                    className="transition-all duration-200 hover:scale-[1.02] active:scale-[0.98]"
                  >
                    <RefreshCw className="mr-2 h-4 w-4" /> Start First Analysis
                  </Button>
                }
              />
            </div>
          )}
        </Card>
      </div>

      {/* MODALS */}
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
