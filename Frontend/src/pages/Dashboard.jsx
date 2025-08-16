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
    Plus,
} from "lucide-react";
import {
    useVideosQuery,
    useVideoStats,
    useUploadVideoMutation,
    useUpdateVideoMutation,
    useDeleteVideoMutation,
} from "../hooks/useVideosQuery.js";
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
import { MODEL_INFO } from "../constants/apiEndpoints.js";
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

export const Dashboard = () => {
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

    // Using a single state object for modals for cleaner management
    const [modal, setModal] = useState({ type: null, data: null });
    const openModal = (type, data = null) => setModal({ type, data });
    const closeModal = () => setModal({ type: null, data: null });

    const [searchTerm, setSearchTerm] = useState("");
    const [statusFilter, setStatusFilter] = useState("ALL");
    const [sizeFilter, setSizeFilter] = useState("ALL");
    const [sortOrder, setSortOrder] = useState("desc");

    const filteredVideos = useMemo(() => {
        let filtered = videos;
        if (searchTerm.trim()) {
            const searchLower = searchTerm.toLowerCase();
            filtered = filtered.filter(
                (video) =>
                    video.filename?.toLowerCase().includes(searchLower) ||
                    video.description?.toLowerCase().includes(searchLower)
            );
        }
        if (statusFilter !== "ALL") {
            filtered = filtered.filter(
                (video) => video.status === statusFilter
            );
        }
        if (sizeFilter !== "ALL") {
            const ranges = {
                small: { min: 0, max: 10485760 },
                medium: { min: 10485760, max: 104857600 },
                large: { min: 104857600, max: Infinity },
            };
            const range = ranges[sizeFilter];
            if (range) {
                filtered = filtered.filter(
                    (v) => v.size >= range.min && v.size < range.max
                );
            }
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
                filterable: true,
                accessor: (item) => item.filename,
            },
            {
                key: "status",
                header: "Status",
                sortable: true,
                filterable: true,
                render: (item) => <StatusBadge status={item.status} />,
            },
            {
                key: "size",
                header: "Size",
                sortable: true,
                filterable: true,
                render: (item) => formatBytes(item.size),
            },
            {
                key: "createdAt",
                header: "Uploaded",
                sortable: true,
                filterable: true,
                render: (item) => formatDate(item.createdAt),
            },
            {
                key: "actions",
                header: "Actions",
                render: (item) => (
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
                                openModal("new_analysis", item);
                            }}
                        >
                            <Brain className="h-5 w-5 text-indigo-500" />
                        </Button>
                        <Button
                            variant="ghost"
                            size="icon"
                            title="Edit Video Details"
                            onClick={(e) => {
                                e.stopPropagation();
                                openModal("edit", item);
                            }}
                        >
                            <Edit className="h-5 w-5 text-blue-500" />
                        </Button>
                        <Button
                            variant="ghost"
                            size="icon"
                            title="Delete Video"
                            onClick={(e) => {
                                e.stopPropagation();
                                openModal("delete", item);
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
                                fetchVideos();
                                showToast.success("Data refreshed!");
                            }}
                            variant="outline"
                        >
                            <RefreshCw className="mr-2 h-5 w-5" /> Refresh
                        </Button>
                        <Button onClick={() => openModal("upload")}>
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
                loading={isAnyLoading}
                emptyMessage="No videos found. Upload your first video to get started!"
                disableInternalSorting={true}
            />

            <UploadModal
                isOpen={modal.type === "upload"}
                onClose={closeModal}
                onUpload={uploadMutation.mutateAsync}
            />
            <EditVideoModal
                isOpen={modal.type === "edit"}
                onClose={closeModal}
                video={modal.data}
                onUpdate={(videoId, data) =>
                    updateMutation.mutateAsync({ videoId, updateData: data })
                }
            />
            <DeleteVideoModal
                isOpen={modal.type === "delete"}
                onClose={closeModal}
                video={modal.data}
                onDelete={deleteMutation.mutateAsync}
            />
            <ModelSelectionModal
                isOpen={modal.type === "new_analysis"}
                onClose={closeModal}
                videoId={modal.data?.id}
                onAnalysisStart={() => {
                    fetchVideos();
                    showToast.info("New analysis has been queued!");
                }}
            />
        </div>
    );
};

export default Dashboard;
