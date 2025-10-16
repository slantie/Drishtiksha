import React, { useState, useMemo, useCallback, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  Upload,
  ShieldX,
  ShieldCheck,
  HelpCircle,
  FileVideo,
  Edit,
  ChartNetwork,
  Trash2,
  FileAudio,
  FileImage,
  RefreshCw,
  Loader2,
  CheckCircle,
  AlertTriangle,
} from "lucide-react";
import { useMediaQuery, useMediaStats } from "../hooks/useMediaQuery.jsx";
import { PageHeader } from "../components/layout/PageHeader.jsx";
import { StatCard } from "../components/ui/StatCard";
import { DataTable } from "../components/ui/DataTable";
import { Button } from "../components/ui/Button";
import { StatusBadge, MediaTypeBadge } from "../components/ui/Badge";
import { UploadModal } from "../components/media/UploadModal.jsx";
import { EditMediaModal } from "../components/media/EditMediaModal.jsx";
import { DeleteMediaModal } from "../components/media/DeleteMediaModal.jsx";
import { MediaSearchFilter } from "../components/media/MediaSearchFilter.jsx";
import { RerunAnalysisModal } from "../components/analysis/RerunAnalysisModal.jsx";
import { formatDate } from "../utils/formatters.js";
import { SkeletonCard } from "../components/ui/SkeletonCard.jsx";

const DashboardSkeleton = () => (
  <div className="space-y-6">
    <SkeletonCard className="h-12" />
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      <SkeletonCard className="h-32" />
      <SkeletonCard className="h-32" />
      <SkeletonCard className="h-32" />
      <SkeletonCard className="h-32" />
    </div>
    <SkeletonCard className="h-80" />
  </div>
);

const MediaTypeIcon = ({ mediaType }) => {
  const iconMap = {
    VIDEO: FileVideo,
    AUDIO: FileAudio,
    IMAGE: FileImage,
  };

  const Icon = iconMap[mediaType] || FileVideo;

  return <MediaTypeBadge mediaType={mediaType} icon={Icon} size="sm" />;
};

const AnalysisSummary = ({ latestRun }) => {
  if (!latestRun || !latestRun.analyses || latestRun.analyses.length === 0) {
    if (latestRun?.status === "FAILED") {
      return (
        <div className="flex items-center gap-2">
          <ShieldX className="w-4 h-4 text-red-500 flex-shrink-0" />
          <div className="flex flex-col">
            <span className="text-sm font-medium text-red-600 dark:text-red-400">
              Failed
            </span>
            <span className="text-xs text-light-muted-text dark:text-dark-muted-text">
              {latestRun.analyses?.length || 0} attempts
            </span>
          </div>
        </div>
      );
    }
    if (latestRun?.status === "PARTIALLY_ANALYZED") {
      const completedCount =
        latestRun.analyses?.filter((a) => a.status === "COMPLETED").length || 0;
      return (
        <div className="flex items-center gap-2">
          <HelpCircle className="w-4 h-4 text-blue-500 flex-shrink-0" />
          <div className="flex flex-col">
            <span className="text-sm font-medium text-blue-600 dark:text-blue-400">
              Partial Results
            </span>
            <span className="text-xs text-light-muted-text dark:text-dark-muted-text">
              {completedCount} models
            </span>
          </div>
        </div>
      );
    }
    return (
      <span className="text-sm text-light-muted-text dark:text-dark-muted-text italic">
        {latestRun?.status === "QUEUED" ? "⏳ Queued" : "⏳ Processing"}...
      </span>
    );
  }

  const completed = latestRun.analyses.filter((a) => a.status === "COMPLETED");
  if (completed.length === 0) {
    return (
      <span className="text-sm text-light-muted-text dark:text-dark-muted-text italic">
        ⏳ Processing... ({latestRun.analyses?.length} started)
      </span>
    );
  }

  const totalFakeScore = completed.reduce((sum, analysis) => {
    const fakeScore =
      analysis.prediction === "FAKE"
        ? analysis.confidence
        : 1 - analysis.confidence;
    return sum + fakeScore;
  }, 0);

  const averageFakeScore = totalFakeScore / completed.length;

  let consensus, Icon, color;
  if (averageFakeScore > 0.55) {
    consensus = "Likely Deepfake";
    Icon = ShieldX;
    color = "text-red-600 dark:text-red-400";
  } else if (averageFakeScore < 0.45) {
    consensus = "Likely Authentic";
    Icon = ShieldCheck;
    color = "text-green-600 dark:text-green-400";
  } else {
    consensus = "Inconclusive";
    Icon = HelpCircle;
    color = "text-yellow-600 dark:text-yellow-400";
  }

  return (
    <div className="flex items-center gap-2">
      <Icon className={`w-4 h-4 ${color} flex-shrink-0`} />
      <div className="flex flex-col">
        <span className={`text-sm font-medium ${color}`}>{consensus}</span>
        <span className="text-xs text-light-muted-text dark:text-dark-muted-text">
          Avg. Fake Score: {(averageFakeScore * 100).toFixed(1)}%
        </span>
      </div>
    </div>
  );
};

export const Dashboard = () => {
  const {
    data: mediaItems = [],
    isLoading,
    isRefetching,
    refetch,
  } = useMediaQuery();
  const { stats } = useMediaStats();
  const navigate = useNavigate();

  useEffect(() => {
    document.title = "Dashboard - Drishtiksha";
  }, []);

  const [modal, setModal] = useState({ type: null, data: null });
  const [filters, setFilters] = useState({
    searchTerm: "",
    status: "ALL",
    mediaType: "ALL",
    prediction: "ALL",
  });

  const onFilterChange = useCallback((newFilters) => {
    setFilters(newFilters);
  }, []);

  const filteredMedia = useMemo(() => {
    return mediaItems
      .filter((item) => {
        if (filters.searchTerm) {
          const searchTermLower = filters.searchTerm.toLowerCase();
          const filenameMatch = item.filename
            .toLowerCase()
            .includes(searchTermLower);
          const descriptionMatch = item.description
            ?.toLowerCase()
            .includes(searchTermLower);
          if (!filenameMatch && !descriptionMatch) return false;
        }

        if (filters.status !== "ALL" && item.status !== filters.status)
          return false;

        if (filters.mediaType !== "ALL" && item.mediaType !== filters.mediaType)
          return false;

        if (filters.prediction !== "ALL") {
          const latestRun = item.analysisRuns?.[0];
          if (!latestRun) return false;

          const completedAnalyses =
            latestRun.analyses?.filter((a) => a.status === "COMPLETED") || [];

          if (completedAnalyses.length === 0) return false;

          const fakes = completedAnalyses.filter(
            (a) => a.prediction === "FAKE"
          ).length;
          const reals = completedAnalyses.length - fakes;

          if (filters.prediction === "FAKE" && fakes <= reals) return false;
          if (filters.prediction === "REAL" && reals <= fakes) return false;
        }
        return true;
      })
      .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
  }, [mediaItems, filters]);

  const columns = useMemo(
    () => [
      {
        key: "mediaType",
        header: "Type",
        render: (item) => <MediaTypeIcon mediaType={item.mediaType} />,
        sortable: true,
        filterable: true,
      },
      {
        key: "filename",
        header: "File",
        render: (item) => (
          <div className="flex flex-col gap-1 max-w-xs">
            <span className="font-semibold text-sm truncate">
              {item.filename}
            </span>
            {item.description && (
              <span className="text-xs text-light-muted-text dark:text-dark-muted-text truncate">
                {item.description}
              </span>
            )}
          </div>
        ),
        sortable: true,
        filterable: true,
      },
      {
        key: "status",
        header: "Job Status",
        render: (item) => <StatusBadge status={item.status} />,
        sortable: true,
        filterable: true,
      },
      {
        key: "summary",
        header: "Latest Result",
        render: (item) => (
          <AnalysisSummary latestRun={item.analysisRuns?.[0]} />
        ),
        sortable: false,
        filterable: true,
      },
      {
        key: "createdAt",
        header: "Uploaded On",
        render: (item) => formatDate(item.createdAt),
        sortable: true,
        filterable: false,
      },
      {
        key: "actions",
        header: "Actions",
        render: (item) => (
          <div className="flex items-center justify-end gap-1">
            <Button
              variant="ghost"
              size="icon"
              title="View Results Page"
              onClick={(e) => {
                e.stopPropagation();
                navigate(`/results/${item.id}`);
              }}
              aria-label={`View results for ${item.filename}`}
            >
              <ChartNetwork className="h-4 w-4 text-purple-500" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              title="Re-run Analysis"
              onClick={(e) => {
                e.stopPropagation();
                setModal({ type: "rerun", data: item });
              }}
              aria-label={`Re-run analysis for ${item.filename}`}
            >
              <RefreshCw className="h-4 w-4 text-indigo-500" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              title="Edit Description"
              onClick={(e) => {
                e.stopPropagation();
                setModal({ type: "edit", data: item });
              }}
              aria-label={`Edit description for ${item.filename}`}
            >
              <Edit className="h-4 w-4 text-blue-500" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              title="Delete Media"
              onClick={(e) => {
                e.stopPropagation();
                setModal({ type: "delete", data: item });
              }}
              aria-label={`Delete ${item.filename}`}
            >
              <Trash2 className="h-4 w-4 text-red-500" />
            </Button>
          </div>
        ),
        sortable: false,
        filterable: false,
      },
    ],
    [navigate]
  );

  if (isLoading && !mediaItems.length) return <DashboardSkeleton />;

  return (
    <div className="space-y-6 w-full max-w-full mx-auto">
      <PageHeader
        title="Media Dashboard"
        description="Manage, upload, and review all your media analyses."
        actions={
          <div className="flex items-center gap-2">
            <Button
              onClick={() => refetch()}
              variant="outline"
              size="sm"
              isLoading={isRefetching}
            >
              <RefreshCw className="mr-2 h-4 w-4" /> Refresh Data
            </Button>
            <Button size="sm" onClick={() => setModal({ type: "upload" })}>
              <Upload className="mr-2 h-4 w-4" /> Upload Media
            </Button>
          </div>
        }
      />
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Files"
          value={stats.total}
          icon={FileVideo}
          isLoading={isLoading}
          cardColor="blue"
        />
        <StatCard
          title="Processing"
          value={stats.processing}
          icon={Loader2}
          isLoading={isLoading}
          cardColor="yellow"
        />
        <StatCard
          title="Analyzed"
          value={stats.analyzed}
          icon={CheckCircle}
          isLoading={isLoading}
          cardColor="green"
        />
        <StatCard
          title="Failed"
          value={stats.failed}
          icon={AlertTriangle}
          isLoading={isLoading}
          cardColor="red"
        />
      </div>
      <MediaSearchFilter
        mediaItems={mediaItems}
        onFilterChange={onFilterChange}
      />
      <DataTable
        columns={columns}
        data={filteredMedia}
        loading={isRefetching}
        title="Media Library"
        description="Your uploaded media files and their analysis statuses."
        onRowClick={(item) => navigate(`/results/${item.id}`)}
        emptyState={{
          icon: FileVideo,
          title: "No Media Found",
          message:
            "Your uploaded files will appear here. Try clearing your filters or uploading a new file.",
          action: (
            <Button onClick={() => setModal({ type: "upload" })}>
              <Upload className="mr-2 h-4 w-4" /> Upload Media
            </Button>
          ),
        }}
      />
      <UploadModal
        isOpen={modal.type === "upload"}
        onClose={() => setModal({ type: null })}
      />
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
        onAnalysisStart={() => refetch()}
      />
    </div>
  );
};

export default Dashboard;
