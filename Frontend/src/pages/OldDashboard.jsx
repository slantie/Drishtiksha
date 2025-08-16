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
    Copy,
    ShieldX,
    ShieldCheck,
    ChartNetwork,
    Brain,
    Activity,
    TrendingUp,
    Plus,
} from "lucide-react";
import {
    useVideosQuery,
    useVideoStats,
    useUploadVideoMutation,
    useUpdateVideoMutation,
    useDeleteVideoMutation,
} from "../hooks/useVideosQuery.js";
import { StatCard } from "../components/ui/StatCard.jsx";
import { DataTable } from "../components/ui/DataTable.jsx";
import { PageLoader } from "../components/ui/LoadingSpinner.jsx";
import { Button } from "../components/ui/Button.jsx";
import { Card } from "../components/ui/Card.jsx";
import { UploadModal } from "../components/videos/UploadModal.jsx";
import { EditVideoModal } from "../components/videos/EditVideoModal.jsx";
import { DeleteVideoModal } from "../components/videos/DeleteVideoModal.jsx";
import { VideoSearchFilter } from "../components/videos/VideoSearchFilter.jsx";
import ModelSelectionModal from "../components/analysis/ModelSelectionModal.jsx";
import { ANALYSIS_TYPE_INFO, MODEL_INFO } from "../constants/apiEndpoints.js";
import { showToast } from "../utils/toast.js";

// Helper functions and components for the DataTable
const formatBytes = (bytes) =>
    bytes ? `${(bytes / 1024 / 1024).toFixed(2)} MB` : "N/A";
const formatDate = (isoDate) => {
    if (!isoDate) return "N/A";
    const date = new Date(isoDate);

    // Format time as HH:MM (24-hour format)
    const timeString = date.toLocaleTimeString("en-US", {
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
    });

    // Format date as DD/MM/YY
    const day = String(date.getDate()).padStart(2, "0");
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const year = String(date.getFullYear()).slice(-2);
    const dateString = `${day}/${month}/${year}`;

    return `${timeString} ${dateString}`;
};

const StatusBadge = ({ status }) => {
    const styles = {
        ANALYZED: "bg-green-500/10 text-green-500",
        PROCESSING: "bg-yellow-500/10 text-yellow-500",
        UPLOADED: "bg-blue-500/10 text-blue-500",
        FAILED: "bg-red-500/10 text-red-500",
    };
    const Icon = {
        ANALYZED: CheckCircle,
        PROCESSING: ProcessingIcon,
        UPLOADED: Clock,
        FAILED: AlertTriangle,
    }[status];
    return (
        <div
            className={`inline-flex items-center gap-2 text-xs font-semibold px-3 py-1.5 rounded-full ${styles[status]}`}
        >
            <Icon
                className={`w-4 h-4 ${
                    status === "PROCESSING" ? "animate-spin" : ""
                }`}
            />
            <span>{status}</span>
        </div>
    );
};

export const Dashboard = () => {
    // TanStack Query hooks
    const {
        data: videos = [],
        isLoading,
        refetch: fetchVideos,
    } = useVideosQuery();
    const { stats } = useVideoStats();
    const uploadMutation = useUploadVideoMutation();
    const updateMutation = useUpdateVideoMutation();
    const deleteMutation = useDeleteVideoMutation();

    const navigate = useNavigate();

    const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
    const [videoToEdit, setVideoToEdit] = useState(null);
    const [videoToDelete, setVideoToDelete] = useState(null);
    const [isModelSelectionOpen, setIsModelSelectionOpen] = useState(false);
    const [selectedVideoForAnalysis, setSelectedVideoForAnalysis] =
        useState(null);

    // Filter states
    const [searchTerm, setSearchTerm] = useState("");
    const [statusFilter, setStatusFilter] = useState("ALL");
    const [sizeFilter, setSizeFilter] = useState("ALL");
    const [sortOrder, setSortOrder] = useState("desc"); // Default to newest first

    // Filtered and sorted videos
    const filteredVideos = useMemo(() => {
        let filtered = videos;

        // Search filter
        if (searchTerm.trim()) {
            const searchLower = searchTerm.toLowerCase();
            filtered = filtered.filter(
                (video) =>
                    video.filename?.toLowerCase().includes(searchLower) ||
                    video.description?.toLowerCase().includes(searchLower)
            );
        }

        // Status filter
        if (statusFilter !== "ALL") {
            filtered = filtered.filter(
                (video) => video.status === statusFilter
            );
        }

        // Size filter
        if (sizeFilter !== "ALL") {
            const sizeRanges = {
                small: { min: 0, max: 10 * 1024 * 1024 },
                medium: { min: 10 * 1024 * 1024, max: 100 * 1024 * 1024 },
                large: { min: 100 * 1024 * 1024, max: Infinity },
            };
            const range = sizeRanges[sizeFilter];
            if (range) {
                filtered = filtered.filter(
                    (video) => video.size >= range.min && video.size < range.max
                );
            }
        }

        // Sort by date (createdAt)
        filtered.sort((a, b) => {
            const aDate = new Date(a.createdAt || 0);
            const bDate = new Date(b.createdAt || 0);
            if (sortOrder === "desc") {
                // Latest first (newest to oldest)
                return bDate.getTime() - aDate.getTime();
            }
            // Oldest first (oldest to newest)
            return aDate.getTime() - bDate.getTime();
        });

        return filtered;
    }, [videos, searchTerm, statusFilter, sizeFilter, sortOrder]);

    const columns = useMemo(
        () => [
            {
                key: "filename",
                header: "File",
                sortable: true, // Disabled - handled by filter
                filterable: true,
                accessor: (item) => item.filename,
            },
            {
                key: "description",
                header: "Description",
                sortable: true, // Disabled - handled by filter
                filterable: true,
                accessor: (item) => item.description,
            },
            {
                key: "status",
                header: "Status",
                sortable: true, // Disabled - handled by filter
                filterable: true,
                accessor: (item) => <StatusBadge status={item.status} />,
            },
            {
                key: "size",
                header: "Size",
                sortable: true, // Disabled - handled by filter
                filterable: true,
                accessor: (item) => item.size,
                render: (item) => formatBytes(item.size),
            },
            {
                key: "createdAt",
                header: "Uploaded",
                sortable: true, // Disabled - handled by filter
                filterable: true,
                accessor: (item) => item.createdAt,
                render: (item) => formatDate(item.createdAt),
            },
            {
                key: "models",
                header: "Analysis Status",
                sortable: true,
                filterable: true,
                accessor: (item) => {
                    const analyses = item.analyses || [];
                    const completedCount = analyses.filter(
                        (a) => a.status === "COMPLETED"
                    ).length;
                    const processingCount = analyses.filter(
                        (a) => a.status === "PROCESSING"
                    ).length;
                    const totalModels = Object.keys(MODEL_INFO).length;

                    if (processingCount > 0)
                        return `${processingCount} Processing`;
                    if (completedCount === 0) return "Not Analyzed";
                    if (completedCount === totalModels) return "Complete";
                    return `${completedCount}/${totalModels} Analyzed`;
                },
                render: (item) => {
                    const analyses = item.analyses || [];
                    const completedCount = analyses.filter(
                        (a) => a.status === "COMPLETED"
                    ).length;
                    const processingCount = analyses.filter(
                        (a) => a.status === "PROCESSING"
                    ).length;
                    const failedCount = analyses.filter(
                        (a) => a.status === "FAILED"
                    ).length;
                    const totalModels = Object.keys(MODEL_INFO).length;

                    const isComplete = completedCount === totalModels;
                    const isProcessing = processingCount > 0;
                    const hasFailures = failedCount > 0;
                    const isPartial =
                        completedCount > 0 && completedCount < totalModels;

                    // Determine most concerning status for color
                    let statusColor, statusIcon;
                    if (isProcessing) {
                        statusColor = "text-yellow-600";
                        statusIcon = (
                            <ProcessingIcon className="h-4 w-4 animate-spin" />
                        );
                    } else if (hasFailures) {
                        statusColor = "text-red-600";
                        statusIcon = <AlertTriangle className="h-4 w-4" />;
                    } else if (isComplete) {
                        statusColor = "text-green-600";
                        statusIcon = <CheckCircle className="h-4 w-4" />;
                    } else if (isPartial) {
                        statusColor = "text-blue-600";
                        statusIcon = <Activity className="h-4 w-4" />;
                    } else {
                        statusColor = "text-gray-500";
                        statusIcon = <Brain className="h-4 w-4" />;
                    }

                    return (
                        <div
                            className={`flex items-center gap-2 ${statusColor}`}
                        >
                            {statusIcon}
                            <div className="flex flex-col">
                                <span className="text-sm font-medium">
                                    {isProcessing &&
                                        `${processingCount} Processing`}
                                    {!isProcessing &&
                                        completedCount === 0 &&
                                        "Not Analyzed"}
                                    {!isProcessing && isComplete && "Complete"}
                                    {!isProcessing &&
                                        isPartial &&
                                        `${completedCount}/${totalModels} Complete`}
                                </span>
                                {(hasFailures || completedCount > 0) && (
                                    <span className="text-xs text-gray-500">
                                        {completedCount > 0 &&
                                            `${completedCount} completed`}
                                        {hasFailures &&
                                            `, ${failedCount} failed`}
                                    </span>
                                )}
                            </div>
                        </div>
                    );
                },
            },
            {
                key: "actions",
                header: "Actions",
                accessor: (item) => (
                    <div className="flex items-center justify-start space-x-2">
                        <Button
                            variant="ghost"
                            size="icon"
                            title="View Analysis Results"
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
                            title="Start New Analysis"
                            onClick={(e) => {
                                e.stopPropagation();
                                setSelectedVideoForAnalysis(item);
                                setIsModelSelectionOpen(true);
                            }}
                            disabled={item.status !== "ANALYZED"}
                        >
                            <Brain className="h-5 w-5 text-indigo-500" />
                        </Button>
                        <Button
                            variant="ghost"
                            size="icon"
                            title="Edit Video Details"
                            onClick={(e) => {
                                e.stopPropagation();
                                setVideoToEdit(item);
                            }}
                        >
                            <Edit className="h-5 w-5 text-blue-500" />
                        </Button>
                        <Button
                            variant="ghost"
                            size="icon"
                            title="Copy Public URL"
                            onClick={(e) => {
                                e.stopPropagation();
                                const publicUrl = item.url.replace(
                                    "/upload/",
                                    "/upload/f_auto,q_auto/"
                                );
                                navigator.clipboard.writeText(publicUrl);
                                showToast.success("Public URL copied!");
                            }}
                        >
                            <Copy className="h-5 w-5 text-gray-500" />
                        </Button>
                        <Button
                            variant="ghost"
                            size="icon"
                            title="Delete Video"
                            onClick={(e) => {
                                e.stopPropagation();
                                setVideoToDelete(item);
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

    // Show loading spinner if any operation is in progress
    const isAnyLoading =
        isLoading ||
        uploadMutation.isPending ||
        updateMutation.isPending ||
        deleteMutation.isPending;

    if (isAnyLoading && !videos.length) {
        return <PageLoader text="Loading Dashboard..." />;
    }

    return (
        <div className="space-y-6 w-full min-h-screen">
            <Card>
                <div className="p-2 flex justify-between items-center">
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
                            onClick={() => {
                                try {
                                    fetchVideos();
                                    showToast.success("Data refreshed!");
                                } catch (error) {
                                    console.error(
                                        "Failed to refresh videos:",
                                        error
                                    );
                                    showToast.error(
                                        "Failed to refresh videos."
                                    );
                                }
                            }}
                            variant="outline"
                        >
                            <RefreshCw className="mr-2 h-5 w-5" /> Refresh
                        </Button>
                        <Button onClick={() => setIsUploadModalOpen(true)}>
                            <Upload className="mr-2 h-5 w-5" /> Upload Video
                        </Button>
                    </div>
                </div>
            </Card>

            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-6">
                <StatCard
                    title="Total Videos"
                    value={stats.total}
                    icon={Play}
                    onClick={() => console.log("Total Videos Clicked")}
                />
                <StatCard
                    title="Total Analyses"
                    value={stats.totalAnalyses}
                    icon={Activity}
                    onClick={() => console.log("Total Analyses Clicked")}
                />
                <StatCard
                    title="Real Detections"
                    value={stats.realDetections}
                    icon={ShieldCheck}
                    onClick={() => console.log("Real Videos Clicked")}
                />
                <StatCard
                    title="Fake Detections"
                    value={stats.fakeDetections}
                    icon={ShieldX}
                    onClick={() => console.log("Fake Videos Clicked")}
                />
            </div>

            {/* Video Search Filter */}
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
                // subtitle={`${filteredVideos.length} of ${videos.length} videos`}
                columns={columns}
                data={filteredVideos}
                onRowClick={(item) => navigate(`/results/${item.id}`)}
                searchPlaceholder="Search videos..."
                showSearch={false}
                loading={isAnyLoading}
                emptyMessage="No videos found. Upload your first video to get started!"
                disableInternalSorting={false}
            />

            <UploadModal
                isOpen={isUploadModalOpen}
                onClose={() => setIsUploadModalOpen(false)}
                onUpload={async (videoFile) => {
                    try {
                        await uploadMutation.mutateAsync(videoFile);
                    } catch (error) {
                        console.error("Upload failed:", error);
                        // Toast handled by the mutation
                    }
                }}
            />
            <EditVideoModal
                isOpen={!!videoToEdit}
                onClose={() => setVideoToEdit(null)}
                video={videoToEdit}
                onUpdate={async (videoId, videoData) => {
                    try {
                        await updateMutation.mutateAsync({
                            videoId,
                            updateData: videoData,
                        });
                        setVideoToEdit(null);
                    } catch (error) {
                        console.error("Update failed:", error);
                        // Toast handled by the mutation
                    }
                }}
            />
            <DeleteVideoModal
                isOpen={!!videoToDelete}
                onClose={() => setVideoToDelete(null)}
                video={videoToDelete}
                onDelete={async (videoId) => {
                    try {
                        await deleteMutation.mutateAsync(videoId);
                        setVideoToDelete(null);
                    } catch (error) {
                        console.error("Delete failed:", error);
                        // Toast handled by the mutation
                    }
                }}
            />
            <ModelSelectionModal
                isOpen={isModelSelectionOpen}
                onClose={() => {
                    setIsModelSelectionOpen(false);
                    setSelectedVideoForAnalysis(null);
                }}
                videoId={selectedVideoForAnalysis?.id}
                onAnalysisStart={() => {
                    fetchVideos(); // Refresh videos to show new analysis status
                    showToast.success("Analysis started successfully!");
                }}
            />
        </div>
    );
};

export default Dashboard;
