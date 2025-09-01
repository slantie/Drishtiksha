// src/pages/DetailedAnalysis.jsx

import React from "react";
import { useParams, Link } from "react-router-dom";
import { ArrowLeft, RefreshCw, AlertTriangle, Video } from "lucide-react";
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
import { showToast } from "../utils/toast.js";
// Correctly import the "smart" report component.
import { AnalysisReport } from "../components/analysis/AnalysisReport.jsx";

const DetailedAnalysis = () => {
  const { mediaId, analysisId } = useParams();
  const {
    data: media,
    isLoading,
    error,
    refetch,
    isRefetching,
  } = useMediaItemQuery(mediaId);

  if (isLoading)
    return <PageLoader text="Loading Detailed Analysis Report..." />;
  if (error)
    return (
      <div className="text-center p-8">
        <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
        <h2 className="text-2xl font-bold">Error Loading Data</h2>
        <p>{error.message}</p>
      </div>
    );
  if (!media) return <PageLoader text="Media not found..." />;

  const analysis = media.analysisRuns
    ?.flatMap((run) => run.analyses)
    .find((a) => a.id === analysisId);

  if (!analysis) {
    return (
      <div className="text-center p-8">
        <AlertTriangle className="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <h2 className="text-2xl font-bold">Analysis Not Found</h2>
        <p>The requested analysis could not be found for this media item.</p>
        <Button asChild variant="outline" className="mt-4">
          <Link to={`/results/${mediaId}`}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Summary
          </Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title={`${analysis.modelName} - Detailed Report`}
        description={`A deep-dive forensic analysis for the file: ${media.filename}`}
        actions={
          <div className="flex items-center gap-2">
            <Button
              onClick={() => refetch()}
              isLoading={isRefetching}
              variant="outline"
            >
              <RefreshCw className="mr-2 h-4 w-4" /> Refresh
            </Button>
            <Button asChild variant="outline">
              <Link to={`/results/${mediaId}`}>
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Summary
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
      <AnalysisReport result={analysis.resultPayload} />
    </div>
  );
};

export default DetailedAnalysis;
