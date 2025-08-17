// src/pages/Dashboard.jsx

import { useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import {
    Upload,
    Play,
    CheckCircle,
    RefreshCw,
    AlertTriangle,
    Edit,
    Trash2,
    Clock,
    Loader2 as ProcessingIcon,
    ShieldX,
    ShieldCheck,
    Activity,
    Brain,
    HelpCircle,
    Loader2,
} from "lucide-react";
import {
    useVideosQuery,
    useVideoStats,
    useUploadVideoMutation,
    useUpdateVideoMutation,
    useDeleteVideoMutation,
} from "../hooks/useVideosQuery.jsx";
import { StatCard } from "../components/ui/StatCard";
import { DataTable } from "../components/ui/DataTable";
import { PageLoader } from "../components/ui/LoadingSpinner";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { UploadModal } from "../components/videos/UploadModal.jsx";
import { EditVideoModal } from "../components/videos/EditVideoModal.jsx";
import { DeleteVideoModal } from "../components/videos/DeleteVideoModal.jsx";
import { VideoSearchFilter } from "../components/videos/VideoSearchFilter.jsx";
import ModelSelectionModal from "../components/analysis/ModelSelectionModal.jsx";
import { showToast } from "../utils/toast.js";
import { formatDate } from "../utils/formatters.js";
import { SkeletonCard } from "../components/ui/SkeletonCard.jsx";

// Page-specific sub-components for cleanliness
const DashboardHeader = ({ onRefresh, isRefetching, onUploadClick }) => {
    const [isManuallyRefetching, setIsManuallyRefetching] = useState(false);

    const handleRefresh = async () => {
        setIsManuallyRefetching(true);
        try {
            await onRefresh();
            showToast.success("Dashboard data has been updated.");
        } catch (error) {
            showToast.error("Failed to refresh data.");
            console.error("Error refreshing dashboard data:", error);
        } finally {
            setIsManuallyRefetching(false);
        }
    };

    return (
        <Card>
            <div className="p-4 flex flex-col sm:flex-row justify-between items-center gap-4">
                <div>
                    <h1 className="text-3xl font-bold">
                        Drishtiksha Dashboard
                    </h1>
                    <p className="text-light-muted-text dark:text-dark-muted-text">
                        Manage your video analyses.
                    </p>
                </div>
                <div className="flex items-center gap-2">
                    <Button
                        onClick={handleRefresh}
                        variant="outline"
                        disabled={isRefetching || isManuallyRefetching}
                        aria-label="Refresh dashboard data"
                    >
                        {isRefetching || isManuallyRefetching ? (
                            <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                        ) : (
                            <RefreshCw className="mr-2 h-5 w-5" />
                        )}
                        {isRefetching || isManuallyRefetching
                            ? "Refreshing..."
                            : "Refresh"}
                    </Button>
                    <Button onClick={onUploadClick}>
                        <Upload className="mr-2 h-5 w-5" /> Upload Video
                    </Button>
                </div>
            </div>
        </Card>
    );
};

const DashboardSkeleton = () => (
    <div className="space-y-6">
        <SkeletonCard className="h-24" />
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-6">
            <SkeletonCard className="h-32" />
            <SkeletonCard className="h-32" />
            <SkeletonCard className="h-32" />
            <SkeletonCard className="h-32" />
        </div>
        <SkeletonCard className="h-24" />
        <SkeletonCard className="h-96" />
    </div>
);

// STATUS BADGE REMAINS THE SAME
const StatusBadge = ({ status }) => {
    const styles = {
        ANALYZED: "bg-green-500/10 text-green-500",
        PARTIALLY_ANALYZED: "bg-blue-500/10 text-blue-500",
        PROCESSING: "bg-yellow-500/10 text-yellow-500",
        QUEUED: "bg-indigo-500/10 text-indigo-500",
        UPLOADED: "bg-gray-500/10 text-gray-500",
        FAILED: "bg-red-500/10 text-red-500",
    };
    const Icon = {
        ANALYZED: CheckCircle,
        PARTIALLY_ANALYZED: CheckCircle,
        PROCESSING: ProcessingIcon,
        QUEUED: Clock,
        UPLOADED: Clock,
        FAILED: AlertTriangle,
    }[status];
    return (
        <div
            className={`inline-flex items-center gap-2 text-xs font-semibold px-3 py-1.5 rounded-full capitalize ${
                styles[status] || styles["UPLOADED"]
            }`}
        >
            <Icon
                className={`w-4 h-4 ${
                    status === "PROCESSING" ? "animate-spin" : ""
                }`}
            />
            <span>{status.replace("_", " ").toLowerCase()}</span>
        </div>
    );
};

// *** NEW ENHANCED ANALYSIS SUMMARY COMPONENT ***
const AnalysisSummary = ({ analyses = [], totalModels = 2 }) => {
    const completed = analyses.filter((a) => a.status === "COMPLETED");
    if (completed.length === 0) {
        return (
            <span className="text-sm text-gray-500 italic">
                Awaiting results...
            </span>
        );
    }

    const fakes = completed.filter((a) => a.prediction === "FAKE").length;
    const reals = completed.filter((a) => a.prediction === "REAL").length;

    let consensus, Icon, color;
    if (fakes > reals) {
        consensus = "Deepfake Detected";
        Icon = ShieldX;
        color = "text-red-500";
    } else if (reals > fakes) {
        consensus = "Authentic";
        Icon = ShieldCheck;
        color = "text-green-500";
    } else {
        consensus = "Inconclusive";
        Icon = HelpCircle;
        color = "text-yellow-500";
    }

    return (
        <div className="flex flex-col">
            <div className={`flex items-center font-bold text-sm ${color}`}>
                <Icon className="w-4 h-4 mr-2 flex-shrink-0" />
                <span>{consensus}</span>
            </div>
            <span className="text-xs text-gray-400 mt-1">
                {completed.length} / {totalModels} Models Complete
            </span>
        </div>
    );
};

export const Dashboard = () => {
    const {
        data: videos = [],
        isLoading,
        refetch,
        isRefetching,
    } = useVideosQuery();
    const { stats } = useVideoStats();
    const uploadMutation = useUploadVideoMutation();
    const updateMutation = useUpdateVideoMutation();
    const deleteMutation = useDeleteVideoMutation();

    const navigate = useNavigate();
    const [modal, setModal] = useState({ type: null, data: null });
    const [searchTerm, setSearchTerm] = useState("");
    const [statusFilter, setStatusFilter] = useState("ALL");
    const [sizeFilter, setSizeFilter] = useState("ALL");
    const [sortOrder, setSortOrder] = useState("desc");

    const filteredVideos = useMemo(() => {
        let filtered = videos;
        // Filtering logic remains the same...
        if (searchTerm.trim()) {
            const searchLower = searchTerm.toLowerCase();
            filtered = filtered.filter(
                (v) =>
                    v.filename?.toLowerCase().includes(searchLower) ||
                    v.description?.toLowerCase().includes(searchLower)
            );
        }
        if (statusFilter !== "ALL") {
            filtered = filtered.filter((v) => v.status === statusFilter);
        }
        if (sizeFilter !== "ALL") {
            const ranges = {
                small: { max: 10485760 },
                medium: { min: 10485760, max: 104857600 },
                large: { min: 104857600 },
            };
            const range = ranges[sizeFilter];
            if (range)
                filtered = filtered.filter(
                    (v) =>
                        (range.min ? v.size >= range.min : true) &&
                        (range.max ? v.size < range.max : true)
                );
        }
        filtered.sort((a, b) => {
            const dateA = new Date(a.createdAt).getTime();
            const dateB = new Date(b.createdAt).getTime();
            return sortOrder === "desc" ? dateB - dateA : dateA - dateB;
        });
        return filtered;
    }, [videos, searchTerm, statusFilter, sizeFilter, sortOrder]);

    const columns = useMemo(
        () => [
            {
                key: "filename",
                header: "File",
                sortable: true,
                accessor: (item) => (
                    <span className="font-medium text-gray-800 dark:text-gray-200">
                        {item.filename}
                    </span>
                ),
            },
            {
                key: "status",
                header: "Job Status",
                sortable: true,
                render: (item) => <StatusBadge status={item.status} />,
            },
            {
                key: "summary",
                header: "Analysis Result",
                render: (item) => <AnalysisSummary analyses={item.analyses} />,
            },
            {
                key: "createdAt",
                header: "Uploaded On",
                sortable: true,
                render: (item) => formatDate(item.createdAt),
            },
            {
                key: "actions",
                header: "Actions",
                render: (item) => (
                    <div className="flex items-center justify-end space-x-1">
                        <Button
                            variant="ghost"
                            size="sm"
                            onClick={(e) => {
                                e.stopPropagation();
                                navigate(`/results/${item.id}`);
                            }}
                        >
                            View Report
                        </Button>
                        <Button
                            variant="ghost"
                            size="icon"
                            title="Delete Video"
                            onClick={(e) => {
                                e.stopPropagation();
                                setModal({ type: "delete", data: item });
                            }}
                        >
                            <Trash2 className="h-4 w-4 text-gray-500" />
                        </Button>
                    </div>
                ),
            },
        ],
        [navigate]
    );

    if (isLoading && !videos.length) return <DashboardSkeleton />;

    return (
        <div className="space-y-6 w-full min-h-screen">
            <DashboardHeader
                onRefresh={refetch}
                isRefetching={isRefetching}
                onUploadClick={() => setModal({ type: "upload" })}
            />

            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-6">
                <StatCard
                    title="Total Videos"
                    value={stats.total}
                    icon={Play}
                />
                <StatCard
                    title="Total Analyses"
                    value={stats.totalAnalyses}
                    icon={Activity}
                />
                <StatCard
                    title="Real Detections"
                    value={stats.realDetections}
                    icon={ShieldCheck}
                />
                <StatCard
                    title="Fake Detections"
                    value={stats.fakeDetections}
                    icon={ShieldX}
                />
            </div>

            <VideoSearchFilter
                searchTerm={searchTerm}
                statusFilter={statusFilter}
                sizeFilter={sizeFilter}
                sortOrder={sortOrder}
                videos={videos}
                onSearchChange={setSearchTerm}
                onStatusFilterChange={setStatusFilter}
                onSizeFilterChange={setSizeFilter}
                onSortOrderChange={setSortOrder}
            />

            <DataTable
                title="Video Library"
                columns={columns}
                data={filteredVideos}
                onRowClick={(item) => navigate(`/results/${item.id}`)}
                searchPlaceholder="Search videos..."
                showSearch={false}
                loading={isRefetching}
                emptyMessage="No videos found. Upload one to get started!"
                disableInternalSorting={true}
            />

            {/* Modals */}
            <UploadModal
                isOpen={modal.type === "upload"}
                onClose={() => setModal({ type: null })}
                onUpload={uploadMutation.mutateAsync}
            />
            <EditVideoModal
                isOpen={modal.type === "edit"}
                onClose={() => setModal({ type: null })}
                video={modal.data}
                onUpdate={(videoId, data) =>
                    updateMutation.mutateAsync({ videoId, updateData: data })
                }
            />
            <DeleteVideoModal
                isOpen={modal.type === "delete"}
                onClose={() => setModal({ type: null })}
                video={modal.data}
                onDelete={deleteMutation.mutateAsync}
            />
            <ModelSelectionModal
                isOpen={modal.type === "new_analysis"}
                onClose={() => setModal({ type: null })}
                videoId={modal.data?.id}
                onAnalysisStart={() => {
                    refetch();
                }}
            />
        </div>
    );
};

export default Dashboard;
