// src/pages/RunDetails.jsx
// Comprehensive view of a single analysis run showing all models combined

import React, { useEffect, useState } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  RefreshCw,
  AlertTriangle,
  Brain,
  SearchX,
  FileText,
  Download,
  BarChart3,
  LineChart,
  PieChart,
  Clock,
  Layers,
  XCircle,
  InfoIcon,
  X,
} from "lucide-react";
import { useMediaItemQuery } from "../hooks/useMediaQuery.jsx";
import { Button } from "../components/ui/Button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "../components/ui/Card";
import { Modal } from "../components/ui/Modal";
import { DataTable } from "../components/ui/DataTable";
import { PageLoader } from "../components/ui/LoadingSpinner";
import { MediaPlayer } from "../components/media/MediaPlayer.jsx";
import { PageHeader } from "../components/layout/PageHeader.jsx";
import { EmptyState } from "../components/ui/EmptyState.jsx";
import { StatusBadge, Badge } from "../components/ui/Badge";
import { formatDate, formatProcessingTime } from "../utils/formatters.js";
import { MarkdownPDFService } from "../services/pdf/MarkdownPDFService.js";
import { showToast } from "../utils/toast.jsx";
import { getModelColor } from "../utils/modelColors.js";

// Import chart components
import { ModelConfidenceChart } from "../components/analysis/charts/ModelConfidenceChart.jsx";
import { PredictionDistributionChart } from "../components/analysis/charts/PredictionDistributionChart.jsx";
import { ProcessingTimeChart } from "../components/analysis/charts/ProcessingTimeChart.jsx";
import { FrameAnalysisChart } from "../components/analysis/charts/FrameAnalysisChart.jsx";
import { RunSummaryCard } from "../components/analysis/charts/RunSummaryCard.jsx";

const RunDetailsPage = () => {
  const { mediaId, runId } = useParams();
  const navigate = useNavigate();
  const [isDownloading, setIsDownloading] = useState(false);
  const [showInfo, setShowInfo] = useState(false);

  const {
    data: media,
    isLoading,
    error,
    refetch,
    isRefetching,
  } = useMediaItemQuery(mediaId);

  useEffect(() => {
    document.title = "Run Details - Drishtiksha";
  }, []);

  // --- Loading State ---
  if (isLoading) return <PageLoader text="Loading Run Details..." />;

  // --- Error Loading Media Data ---
  if (error)
    return (
      <div className="text-center p-8 w-full max-w-full mx-auto space-y-4">
        <AlertTriangle className="w-16 h-16 text-red-500 mx-auto" />
        <h2 className="text-2xl font-bold">Error Loading Run Details</h2>
        <p className="text-light-muted-text dark:text-dark-muted-text">
          {error.message ||
            "An unexpected error occurred while fetching run details."}
        </p>
        <Button
          onClick={() => refetch()}
          variant="outline"
          className="mt-4"
          aria-label="Try again to load run details"
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
                <ArrowLeft className="mr-2 h-4 w-4" />
                Go Back
              </Link>
            </Button>
          }
        />
      </div>
    );

  // Find the specific run
  const run = media.analysisRuns?.find((r) => r.id === runId);

  // --- Run Not Found ---
  if (!run) {
    return (
      <div className="text-center p-8 w-full max-w-full mx-auto space-y-4">
        <EmptyState
          icon={SearchX}
          title="Analysis Run Not Found"
          message="The requested analysis run could not be found."
          action={
            <Button asChild aria-label="Go Back">
              <Link to={`/results/${mediaId}`}>
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Results
              </Link>
            </Button>
          }
        />
      </div>
    );
  }

  const handleDownloadReport = async () => {
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

  // Calculate statistics
  const completedAnalyses =
    run.analyses?.filter((a) => a.status === "COMPLETED") || [];
  const failedAnalyses =
    run.analyses?.filter((a) => a.status === "FAILED") || [];

  const realPredictions = completedAnalyses.filter(
    (a) => a.prediction === "REAL"
  ).length;
  const fakePredictions = completedAnalyses.filter(
    (a) => a.prediction === "FAKE"
  ).length;

  const avgConfidence =
    completedAnalyses.length > 0
      ? completedAnalyses.reduce((sum, a) => sum + (a.confidence || 0), 0) /
        completedAnalyses.length
      : 0;

  // Get all model names for consistent color mapping across all charts
  const allModelNames = run.analyses?.map((a) => a.modelName) || [];

  return (
    <div className="w-full max-w-full mx-auto space-y-6">
      {/* Information Modal */}
      <Modal
        isOpen={showInfo}
        onClose={() => setShowInfo(false)}
        title="Temporal Analysis Information"
        description="Understanding the frame-by-frame analysis visualization"
        size="md"
      >
        <div className="space-y-4">
          <div className="flex items-start gap-3 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
            <InfoIcon className="h-5 w-5 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
            <div className="space-y-2">
              <p className="text-sm text-light-text dark:text-dark-text">
                <strong>Normalized Timeline:</strong> All model predictions are
                normalized and displayed on the same timeline for easy
                comparison, regardless of their original sampling rates.
              </p>
            </div>
          </div>

          <div className="space-y-3 text-sm text-light-muted-text dark:text-dark-muted-text">
            <div className="flex items-start gap-2">
              <span className="text-lg">📊</span>
              <p>
                <strong>Score Interpretation:</strong> Higher scores indicate
                higher likelihood of manipulation. Scores are normalized to a
                0-1 scale.
              </p>
            </div>

            <div className="flex items-start gap-2">
              <span className="text-lg">🔄</span>
              <p>
                <strong>Interpolation:</strong> Data is interpolated for models
                with different sampling rates to provide smooth visualization
                and accurate comparisons.
              </p>
            </div>

            <div className="flex items-start gap-2">
              <span className="text-lg">🎨</span>
              <p>
                <strong>Color Legend:</strong> Each line represents a different
                model. Use the legend to identify which model each line
                represents.
              </p>
            </div>
          </div>
        </div>
      </Modal>

      {/* Page Header */}
      <PageHeader
        title={`Run #${run.runNumber} - ${media.filename}`}
        description={`Comprehensive analysis run performed on ${formatDate(
          run.createdAt
        )}`}
        icon={Layers}
        actions={
          <div className="flex flex-wrap gap-3">
            <Button
              onClick={handleDownloadReport}
              variant="outline"
              disabled={isDownloading}
              aria-label="Download PDF Report"
            >
              {isDownloading ? (
                <>
                  <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Download className="h-4 w-4 mr-2" />
                  Download Report
                </>
              )}
            </Button>
            <Button asChild variant="outline">
              <Link to={`/results/${mediaId}`}>
                <ArrowLeft className="mr-2 h-4 w-4" />
                Go Back
              </Link>
            </Button>
          </div>
        }
      />

      <div className="grid grid-cols-3 gap-6">
        {/* Media Player */}
        <div className="col-span-1">
          {media?.url && (
            <Card className="p-0">
              <CardContent className="p-0">
                <MediaPlayer media={media} />
              </CardContent>
            </Card>
          )}
        </div>

        <div className="col-span-2 h-full">
          {/* Run Summary Card */}
          <RunSummaryCard
            completedCount={completedAnalyses.length}
            failedCount={failedAnalyses.length}
            realCount={realPredictions}
            fakeCount={fakePredictions}
            avgConfidence={avgConfidence}
          />
        </div>
      </div>

      {/* Frame-Level Analysis (for models with frame data) */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <LineChart className="h-5 w-5" />
              Temporal Analysis
            </CardTitle>
            <CardDescription>
              Frame-by-frame confidence trends over time
            </CardDescription>
          </div>
          <div>
            <Button
              className="px-2 py-1"
              variant="ghost"
              size="sm"
              onClick={() => setShowInfo(true)}
              aria-label="Information about frame-level analysis"
            >
              <InfoIcon className="h-5 w-5" />
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <FrameAnalysisChart analyses={completedAnalyses} />
        </CardContent>
      </Card>

      <div className="grid grid-cols-2 w-full gap-6">
        {/* Prediction Distribution
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <PieChart className="h-5 w-5" />
              Prediction Distribution
            </CardTitle>
            <CardDescription>
              Distribution of REAL vs FAKE classifications
            </CardDescription>
          </CardHeader>
          <CardContent>
            <PredictionDistributionChart
              realCount={realPredictions}
              fakeCount={fakePredictions}
            />
          </CardContent>
        </Card> */}

        {/* Model Confidence Comparison */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Model Confidence Comparison
            </CardTitle>
            <CardDescription>
              Confidence levels (%) across all detection models
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ModelConfidenceChart analyses={completedAnalyses} />
          </CardContent>
        </Card>

        {/* Processing Time Analysis */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Clock className="h-5 w-5" />
              Processing Time Analysis
            </CardTitle>
            <CardDescription>
              Time taken (s) by each model to complete analysis
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ProcessingTimeChart analyses={completedAnalyses} />
          </CardContent>
        </Card>
      </div>

      {/* Individual Model Results */}
      <DataTable
        data={run.analyses || []}
        columns={[
          {
            key: "modelName",
            header: "Model",
            sortable: true,
            render: (item) => (
              <div className="flex items-center gap-2">
                <div
                  className="w-3 h-3 rounded-full flex-shrink-0"
                  style={{
                    backgroundColor: getModelColor(
                      item.modelName,
                      allModelNames
                    ),
                  }}
                  title={`Color indicator for ${item.modelName}`}
                />
                <div className="font-medium">{item.modelName}</div>
              </div>
            ),
          },
          {
            key: "status",
            header: "Status",
            sortable: true,
            render: (item) => <StatusBadge status={item.status} size="sm" />,
          },
          {
            key: "prediction",
            header: "Prediction",
            sortable: true,
            render: (item) =>
              item.status === "COMPLETED" ? (
                <Badge
                  variant={item.prediction === "REAL" ? "success" : "danger"}
                  size="sm"
                >
                  {item.prediction}
                </Badge>
              ) : (
                <span className="text-light-muted-text dark:text-dark-muted-text text-sm">
                  N/A
                </span>
              ),
          },
          {
            key: "confidence",
            header: "Confidence",
            align: "right",
            sortable: true,
            render: (item) =>
              item.status === "COMPLETED" && item.confidence ? (
                <span className="font-mono text-sm">
                  {(item.confidence * 100).toFixed(2)}%
                </span>
              ) : (
                <span className="text-light-muted-text dark:text-dark-muted-text text-sm">
                  N/A
                </span>
              ),
          },
          {
            key: "processingTime",
            header: "Processing Time",
            align: "right",
            sortable: true,
            render: (item) =>
              item.processingTime ? (
                <span className="text-sm">
                  {formatProcessingTime(item.processingTime)}
                </span>
              ) : (
                <span className="text-light-muted-text dark:text-dark-muted-text text-sm">
                  N/A
                </span>
              ),
          },
          {
            key: "actions",
            header: "Actions",
            align: "right",
            render: (item) =>
              item.status === "COMPLETED" ? (
                <Button asChild variant="outline" size="sm">
                  <Link to={`/results/${mediaId}/${item.id}`}>
                    <FileText className="h-4 w-4 mr-1" />
                    View Details
                  </Link>
                </Button>
              ) : null,
          },
        ]}
        loading={isLoading}
        onRowClick={(analysis) =>
          analysis.status === "COMPLETED" &&
          navigate(`/results/${mediaId}/${analysis.id}`)
        }
        showPagination={false}
        showSearch={false}
        title={
          <div className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Individual Model Results
          </div>
        }
        description="Detailed results from each detection model"
        emptyState={{
          icon: Brain,
          title: "No Analyses Found",
          message: "This run does not have any model analyses yet.",
        }}
      />

      {/* Failed Analyses Warning */}
      {failedAnalyses.length > 0 && (
        <Card className="border-l-4 border-l-amber-500">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-amber-600 dark:text-amber-400">
              <AlertTriangle className="h-5 w-5" />
              Failed Analyses
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-light-muted-text dark:text-dark-muted-text mb-3">
              {failedAnalyses.length} model(s) failed during analysis. This may
              indicate:
            </p>
            <ul className="list-disc list-inside text-sm text-light-muted-text dark:text-dark-muted-text space-y-1 mb-4">
              <li>
                Missing required media features (e.g., audio track for
                audio-visual models)
              </li>
              <li>Incompatible media format or encoding</li>
              <li>Technical issues during processing</li>
            </ul>
            <div className="flex flex-wrap gap-2">
              {failedAnalyses.map((analysis) => (
                <Badge key={analysis.id} variant="secondary">
                  <XCircle className="h-3 w-3 mr-1" />
                  {analysis.modelName}
                </Badge>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default RunDetailsPage;
