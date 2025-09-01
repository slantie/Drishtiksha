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
} from "lucide-react";
import {
  useMediaItemQuery,
  useDeleteMediaMutation,
} from "../hooks/useMediaQuery.jsx";
import { Button } from "../components/ui/Button";
import {
  Card,
  CardContent,
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

const AnalysisRunCard = ({ run, mediaId }) => (
  <Card>
    <CardHeader>
      <div className="flex justify-between items-center">
        <CardTitle>Analysis Run #{run.runNumber}</CardTitle>
        <span className="text-xs text-light-muted-text dark:text-dark-muted-text">
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
        {run.analyses?.map((analysis) => (
          <AnalysisResultSummaryCard
            key={analysis.id}
            analysis={analysis}
            mediaId={mediaId}
          />
        ))}
      </div>
    </CardContent>
  </Card>
);

const Results = () => {
  const { mediaId } = useParams();
  const {
    data: media,
    isLoading,
    error,
    refetch,
  } = useMediaItemQuery(mediaId, {
    refetchInterval: (query) =>
      ["QUEUED", "PROCESSING"].includes(
        query.state.data?.status || query.state.data?.analysisRuns?.[0]?.status
      )
        ? 5000
        : false,
  });
  const deleteMutation = useDeleteMediaMutation();
  const [modal, setModal] = useState({ type: null, data: null });

  if (isLoading) return <PageLoader text="Loading Media & Results..." />;
  if (error)
    return (
      <div className="text-center p-8">
        <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
        <h2 className="text-2xl font-bold">Error Loading Media</h2>
        <p>{error.message}</p>
      </div>
    );
  if (!media) return <PageLoader text="Media not found..." />;

  const latestRun = media.analysisRuns?.[0];
  const isProcessing = ["QUEUED", "PROCESSING"].includes(latestRun?.status);

  return (
    <div className="space-y-6">
      <PageHeader
        title={media.filename}
        description={media.description || "Analysis results and history."}
        actions={
          <Button asChild variant="outline">
            <Link to="/dashboard">
              <ArrowLeft className="mr-2 h-4 w-4" /> Back to Dashboard
            </Link>
          </Button>
        }
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
        <div className="lg:col-span-1 space-y-6 sticky top-24">
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
            <CardContent className="grid grid-cols-2 gap-2">
              <Button onClick={() => setModal({ type: "rerun", data: media })}>
                <RefreshCw className="h-4 w-4 mr-2" /> Re-Analyze
              </Button>
              <Button
                onClick={() => setModal({ type: "edit", data: media })}
                variant="outline"
              >
                <Edit className="h-4 w-4 mr-2" /> Edit
              </Button>
              <Button
                onClick={() => window.open(media.url, "_blank")}
                variant="outline"
              >
                <Download className="h-4 w-4 mr-2" /> Download File
              </Button>
              <Button
                onClick={() => setModal({ type: "delete", data: media })}
                variant="destructive"
              >
                <Trash2 className="h-4 w-4 mr-2" /> Delete
              </Button>
            </CardContent>
          </Card>
        </div>

        <div className="lg:col-span-2 space-y-6">
          {isProcessing ? (
            <AnalysisInProgress media={media} />
          ) : media.analysisRuns && media.analysisRuns.length > 0 ? (
            <>
              <div className="flex items-center gap-2 text-light-muted-text dark:text-dark-muted-text">
                <Layers className="h-5 w-5" />
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
              message="This media is awaiting its first analysis."
              action={
                <Button
                  onClick={() => setModal({ type: "rerun", data: media })}
                >
                  <RefreshCw className="mr-2 h-4 w-4" /> Start First Analysis
                </Button>
              }
            />
          )}
        </div>
      </div>

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
