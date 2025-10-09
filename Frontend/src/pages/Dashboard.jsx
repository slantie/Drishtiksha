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
  Trash2,
  FileAudio,
  FileImage,
  RefreshCw,
  Loader2, // Added Loader2 for processing stat card
  CheckCircle, // Added CheckCircle for analyzed stat card
  AlertTriangle, // Added AlertTriangle for failed stat card
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
import { MediaSearchFilter } from "../components/media/MediaSearchFilter.jsx";
import { RerunAnalysisModal } from "../components/analysis/RerunAnalysisModal.jsx";
import { formatDate } from "../utils/formatters.js";
import { SkeletonCard } from "../components/ui/SkeletonCard.jsx";
// Removed useAuth as it's not directly consumed here.

// --- SUB-COMPONENTS (Refined for Media Types) ---

const DashboardSkeleton = () => (
  <div className="space-y-6 max-w-full mx-auto">
    {" "}
    {/* Adjusted spacing and full width */}
    <SkeletonCard className="h-24 w-full" /> {/* PageHeader skeleton */}
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
      {" "}
      {/* Consistent gap */}
      <SkeletonCard className="h-32" />
      <SkeletonCard className="h-32" />
      <SkeletonCard className="h-32" />
      <SkeletonCard className="h-32" />
    </div>
    <SkeletonCard className="h-24 w-full" /> {/* Filter card skeleton */}
    <SkeletonCard className="h-[500px] w-full" /> {/* DataTable skeleton */}
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
  if (!latestRun || !latestRun.analyses || latestRun.analyses.length === 0) {
    // If the run status is already failed or partially analyzed (but no completed analyses)
    if (latestRun?.status === "FAILED") {
      return (
        <div className={`flex items-center font-semibold text-sm text-red-500`}>
          <ShieldX className="w-5 h-5 mr-2 flex-shrink-0" />
          <span>Failed ({latestRun.analyses?.length || 0} attempts)</span>
        </div>
      );
    }
    if (latestRun?.status === "PARTIALLY_ANALYZED") {
      return (
        <div
          className={`flex items-center font-semibold text-sm text-blue-500`}
        >
          <HelpCircle className="w-5 h-5 mr-2 flex-shrink-0" />
          <span>
            Partial Results (
            {latestRun.analyses?.filter((a) => a.status === "COMPLETED").length}{" "}
            models)
          </span>
        </div>
      );
    }
    // For QUEUED/PROCESSING, or if no analyses yet
    return (
      <span className="text-sm text-light-muted-text dark:text-dark-muted-text italic">
        {latestRun?.status === "QUEUED" ? "Queued" : "Processing"}...
      </span>
    );
  }

  const completed = latestRun.analyses.filter((a) => a.status === "COMPLETED");
  if (completed.length === 0) {
    return (
      <span className="text-sm text-light-muted-text dark:text-dark-muted-text italic">
        Processing... ({latestRun.analyses?.length} job(s) started)
      </span>
    );
  }

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
  const {
    data: mediaItems = [],
    isLoading,
    isRefetching,
    refetch,
  } = useMediaQuery(); // Added refetch from useMediaQuery
  const { stats } = useMediaStats();
  const navigate = useNavigate();

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
        // Filter by search term (filename or description)
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

        // Filter by media status
        if (filters.status !== "ALL" && item.status !== filters.status)
          return false;

        // Filter by media type
        if (filters.mediaType !== "ALL" && item.mediaType !== filters.mediaType)
          return false;

        // Filter by prediction consensus (from latest run)
        if (filters.prediction !== "ALL") {
          const latestRun = item.analysisRuns?.[0];
          if (!latestRun) return false; // If no runs, cannot match prediction filter

          const completedAnalyses =
            latestRun.analyses?.filter((a) => a.status === "COMPLETED") || [];

          if (completedAnalyses.length === 0) return false; // No completed analyses to determine prediction

          const fakes = completedAnalyses.filter(
            (a) => a.prediction === "FAKE"
          ).length;
          const reals = completedAnalyses.length - fakes;

          if (filters.prediction === "FAKE" && fakes <= reals) return false;
          if (filters.prediction === "REAL" && reals <= fakes) return false;
        }
        return true;
      })
      .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt)); // Always sort by most recent
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
          <span className="font-semibold">{item.filename}</span>
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
        sortable: false, // This is a complex rendered component, not easily sortable
        filterable: true, // Filterable by prediction via MediaSearchFilter
      },
      {
        key: "createdAt",
        header: "Uploaded On",
        render: (item) => formatDate(item.createdAt),
        sortable: true,
        filterable: false, // Dates usually filtered by range, not simple search
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
              aria-label={`View results for ${item.filename}`}
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
              aria-label={`Re-run analysis for ${item.filename}`}
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
              aria-label={`Edit description for ${item.filename}`}
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
              aria-label={`Delete ${item.filename}`}
            >
              <Trash2 className="h-5 w-5 text-red-500" />
            </Button>
          </div>
        ),
        sortable: false,
        filterable: false,
      },
    ],
    [navigate]
  );

  // If initial loading or no media items at all
  if (isLoading && !mediaItems.length) return <DashboardSkeleton />;

  return (
    <div className="space-y-2 w-full max-w-full mx-auto">
      {" "}
      {/* Consistent vertical spacing, full width */}
      <PageHeader
        title="Media Dashboard"
        description="Manage, upload, and review all your media analyses."
        actions={
          <div className="flex items-center gap-2">
            {" "}
            {/* Group buttons */}
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
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
        {" "}
        {/* Consistent gap */}
        <StatCard
          title="Total Files"
          value={stats.total}
          icon={FileVideo} // More generic icon for total files
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
        {/* You can still keep these if you want to differentiate total analyses vs total files by status */}
        {/*
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
        */}
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
        // Removed onDelete prop as it's handled by the hook itself
      />
      <RerunAnalysisModal
        isOpen={modal.type === "rerun"}
        onClose={() => setModal({ type: null })}
        media={modal.data}
        onAnalysisStart={() => refetch()} // Pass refetch to update media data after rerun
      />
    </div>
  );
};

export default Dashboard;
