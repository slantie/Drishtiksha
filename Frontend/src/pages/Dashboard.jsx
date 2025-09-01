// src/pages/Dashboard.jsx

import React, { useState, useMemo, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import {
  Upload,
  Play,
  ShieldX,
  ShieldCheck,
  HelpCircle,
  FileVideo,
  Edit,
  ChartNetwork,
  Download,
  Trash2,
  FileAudio,
  FileImage,
  RefreshCw,
} from "lucide-react";
import {
  useMediaQuery,
  useMediaStats,
  useDeleteMediaMutation,
} from "../hooks/useMediaQuery.jsx";
import { PageHeader } from "../components/layout/PageHeader.jsx";
import { StatCard } from "../components/ui/StatCard";
import { DataTable } from "../components/ui/DataTable";
import { Button } from "../components/ui/Button";
import { UploadModal } from "../components/media/UploadModal.jsx";
import { EditMediaModal } from "../components/media/EditMediaModal.jsx";
import { DeleteMediaModal } from "../components/media/DeleteMediaModal.jsx";
// REFACTOR: Import our new, powerful filter and rerun modal components.
import { MediaSearchFilter } from "../components/media/MediaSearchFilter.jsx";
import { RerunAnalysisModal } from "../components/analysis/RerunAnalysisModal.jsx";
import { formatDate } from "../utils/formatters.js";
import { SkeletonCard } from "../components/ui/SkeletonCard.jsx";
import { useAuth } from "../contexts/AuthContext.jsx";

// --- SUB-COMPONENTS (Refined for Media Types) ---

const DashboardSkeleton = () => (
  <div className="space-y-8">
    <SkeletonCard className="h-12 w-full" />
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
      <SkeletonCard className="h-32" /> <SkeletonCard className="h-32" />
      <SkeletonCard className="h-32" /> <SkeletonCard className="h-32" />
    </div>
    <SkeletonCard className="h-96" />
  </div>
);

const StatusBadge = ({ status }) => {
  const styles = {
    ANALYZED: "bg-green-500/10 text-green-500",
    PARTIALLY_ANALYZED: "bg-blue-500/10 text-blue-500",
    PROCESSING: "bg-yellow-500/10 text-yellow-500 animate-pulse",
    QUEUED: "bg-indigo-500/10 text-indigo-500",
    FAILED: "bg-red-500/10 text-red-500",
  };
  return (
    <div
      className={`inline-flex items-center gap-2 text-xs font-semibold px-3 py-1 rounded-full capitalize ${
        styles[status] || "bg-gray-500/10 text-gray-500"
      }`}
    >
      <span>{status.replace("_", " ").toLowerCase()}</span>
    </div>
  );
};

const MediaTypeIcon = ({ mediaType }) => {
  switch (mediaType) {
    case "VIDEO":
      return <FileVideo className="w-5 h-5 text-purple-500" />;
    case "AUDIO":
      return <FileAudio className="w-5 h-5 text-blue-500" />;
    case "IMAGE":
      return <FileImage className="w-5 h-5 text-green-500" />;
    default:
      return <FileVideo className="w-5 h-5 text-gray-500" />;
  }
};

const AnalysisSummary = ({ latestRun }) => {
  if (!latestRun || !latestRun.analyses) {
    return (
      <span className="text-sm text-light-muted-text dark:text-dark-muted-text italic">
        Awaiting results...
      </span>
    );
  }
  const completed = latestRun.analyses.filter((a) => a.status === "COMPLETED");
  if (completed.length === 0)
    return (
      <span className="text-sm text-light-muted-text dark:text-dark-muted-text italic">
        Processing...
      </span>
    );

  const fakes = completed.filter((a) => a.prediction === "FAKE").length;
  const reals = completed.length - fakes;

  let consensus, Icon, color;
  if (fakes > reals) {
    consensus = "Likely Deepfake";
    Icon = ShieldX;
    color = "text-red-500";
  } else if (reals > fakes) {
    consensus = "Likely Authentic";
    Icon = ShieldCheck;
    color = "text-green-500";
  } else {
    consensus = "Inconclusive";
    Icon = HelpCircle;
    color = "text-yellow-500";
  }

  return (
    <div className={`flex items-center font-semibold text-sm ${color}`}>
      <Icon className="w-5 h-5 mr-2 flex-shrink-0" />
      <div>
        {consensus}
        <span className="block text-xs font-normal text-light-muted-text dark:text-dark-muted-text">
          {completed.length} Model(s) Analyzed
        </span>
      </div>
    </div>
  );
};

// --- MAIN DASHBOARD PAGE ---

export const Dashboard = () => {
  const { data: mediaItems = [], isLoading, isRefetching } = useMediaQuery();
  const { stats } = useMediaStats();
  const deleteMutation = useDeleteMediaMutation();
  const navigate = useNavigate();

  const [modal, setModal] = useState({ type: null, data: null });
  // REFACTOR: The filter state is now a single, clean object.
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
        if (
          filters.searchTerm &&
          !item.filename
            .toLowerCase()
            .includes(filters.searchTerm.toLowerCase())
        )
          return false;
        if (filters.status !== "ALL" && item.status !== filters.status)
          return false;
        if (filters.mediaType !== "ALL" && item.mediaType !== filters.mediaType)
          return false;

        if (filters.prediction !== "ALL") {
          const latestRun = item.analysisRuns?.[0];
          if (!latestRun) return false;
          const fakes = latestRun.analyses.filter(
            (a) => a.prediction === "FAKE" && a.status === "COMPLETED"
          ).length;
          const reals = latestRun.analyses.filter(
            (a) => a.prediction === "REAL" && a.status === "COMPLETED"
          ).length;
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
      },
      {
        key: "filename",
        header: "File",
        render: (item) => (
          <span className="font-semibold">{item.filename}</span>
        ),
      },
      {
        key: "status",
        header: "Job Status",
        render: (item) => <StatusBadge status={item.status} />,
      },
      // REFACTOR: Pass the latest analysis run to the summary component.
      {
        key: "summary",
        header: "Latest Result",
        render: (item) => (
          <AnalysisSummary latestRun={item.analysisRuns?.[0]} />
        ),
      },
      {
        key: "createdAt",
        header: "Uploaded On",
        render: (item) => formatDate(item.createdAt),
      },
      {
        key: "actions",
        header: "Actions",
        render: (item) => (
          <div className="flex items-center justify-end space-x-1">
            <Button
              variant="ghost"
              size="icon"
              title="View Results Page"
              onClick={(e) => {
                e.stopPropagation();
                navigate(`/results/${item.id}`);
              }}
            >
              <ChartNetwork className="h-5 w-5 text-purple-500" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              title="Re-run Analysis"
              onClick={(e) => {
                e.stopPropagation();
                setModal({ type: "rerun", data: item });
              }}
            >
              <RefreshCw className="h-5 w-5 text-indigo-500" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              title="Edit Description"
              onClick={(e) => {
                e.stopPropagation();
                setModal({ type: "edit", data: item });
              }}
            >
              <Edit className="h-5 w-5 text-blue-500" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              title="Delete Media"
              onClick={(e) => {
                e.stopPropagation();
                setModal({ type: "delete", data: item });
              }}
            >
              <Trash2 className="h-5 w-5 text-red-500" />
            </Button>
          </div>
        ),
      },
    ],
    [navigate]
  );

  if (isLoading && !mediaItems.length) return <DashboardSkeleton />;

  return (
    <div className="space-y-4">
      <PageHeader
        title="Media Dashboard"
        description="Manage, upload, and review all your media analyses."
        actions={
          <Button onClick={() => setModal({ type: "upload" })}>
            <Upload className="mr-2 h-4 w-4" /> Upload Media
          </Button>
        }
      />

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Files"
          value={stats.total}
          icon={Play}
          isLoading={isLoading}
          cardColor="blue"
        />
        <StatCard
          title="Total Analyses"
          value={stats.totalAnalyses}
          icon={ChartNetwork}
          isLoading={isLoading}
          cardColor="purple"
        />
        <StatCard
          title="Authentic Results"
          value={stats.realDetections}
          icon={ShieldCheck}
          isLoading={isLoading}
          cardColor="green"
        />
        <StatCard
          title="Deepfake Results"
          value={stats.fakeDetections}
          icon={ShieldX}
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

      {/* --- MODALS --- */}
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
        onDelete={deleteMutation.mutateAsync}
      />
      <RerunAnalysisModal
        isOpen={modal.type === "rerun"}
        onClose={() => setModal({ type: null })}
        media={modal.data}
      />
    </div>
  );
};

export default Dashboard;
