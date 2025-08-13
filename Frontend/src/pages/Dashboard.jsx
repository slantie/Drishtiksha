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
} from "lucide-react";
import { useVideos } from "../hooks/useVideos.js";
import { StatCard } from "../components/ui/StatCard";
import { DataTable } from "../components/ui/DataTable";
import { PageLoader } from "../components/ui/LoadingSpinner";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { UploadModal } from "../components/videos/UploadModal.jsx";
import { EditVideoModal } from "../components/videos/EditVideoModal.jsx";
import { DeleteVideoModal } from "../components/videos/DeleteVideoModal.jsx";
import { showToast } from "../utils/toast.js";

// Helper functions and components for the DataTable
const formatBytes = (bytes) =>
    bytes ? `${(bytes / 1024 / 1024).toFixed(2)} MB` : "N/A";
const formatDate = (isoDate) =>
    new Date(isoDate).toLocaleDateString("en-US", {
        year: "numeric",
        month: "short",
        day: "numeric",
    });

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
    const {
        videos,
        isLoading,
        stats,
        fetchVideos,
        uploadVideo,
        updateVideo,
        deleteVideo,
    } = useVideos();
    const navigate = useNavigate();

    const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
    const [videoToEdit, setVideoToEdit] = useState(null);
    const [videoToDelete, setVideoToDelete] = useState(null);

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
                key: "description",
                header: "Description",
                sortable: true,
                filterable: true,
                accessor: (item) => item.description,
            },
            {
                key: "status",
                header: "Status",
                sortable: true,
                filterable: true,
                accessor: (item) => <StatusBadge status={item.status} />,
            },
            {
                key: "size",
                header: "Size",
                sortable: true,
                filterable: true,
                accessor: (item) => item.size,
                render: (item) => formatBytes(item.size),
            },
            {
                key: "createdAt",
                header: "Uploaded",
                sortable: true,
                filterable: true,
                accessor: (item) => item.createdAt,
                render: (item) => formatDate(item.createdAt),
            },
            {
                key: "models",
                header: "Models",
                sortable: true,
                filterable: true,
                accessor: (item) => {
                    const totalModels = 3; // Expected total models
                    const completedAnalyses = item.analyses
                        ? item.analyses.filter(
                              (analysis) => analysis.status === "COMPLETED"
                          ).length
                        : 0;
                    return `${completedAnalyses}/${totalModels}`;
                },
                render: (item) => {
                    const totalModels = 3; // Expected total models
                    const completedAnalyses = item.analyses
                        ? item.analyses.filter(
                              (analysis) => analysis.status === "COMPLETED"
                          ).length
                        : 0;
                    const isComplete = completedAnalyses === totalModels;
                    const isPartial =
                        completedAnalyses > 0 &&
                        completedAnalyses < totalModels;

                    return (
                        <div
                            className={`${
                                isComplete
                                    ? "text-green-500"
                                    : isPartial
                                    ? "text-yellow-500"
                                    : "text-gray-500"
                            }`}
                        >
                            <span>
                                {completedAnalyses}/{totalModels} Models
                                Analyzed
                            </span>
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
                            <ChartNetwork className="h-4 w-4" />
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
                            <Copy className="h-4 w-4" />
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
                            <Edit className="h-4 w-4" />
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
                            <Trash2 className="h-4 w-4 text-red-500" />
                        </Button>
                    </div>
                ),
            },
        ],
        [navigate, setVideoToEdit, setVideoToDelete]
    );

    if (isLoading) {
        return <PageLoader text="Loading Dashboard..." />;
    }

    return (
        <div className="space-y-6 w-full min-h-screen">
            <Card>
                <div className="p-2 flex justify-between items-center">
                    <div>
                        <h1 className="text-4xl font-bold">
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
                                    showToast.success(
                                        "Videos Data refreshed successfully!"
                                    );
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
                    title="Videos Analyzed"
                    value={stats.analyzed}
                    icon={ChartNetwork}
                    onClick={() => console.log("Analyzed Videos Clicked")}
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

            <DataTable
                columns={columns}
                data={videos}
                onRowClick={(item) => navigate(`/results/${item.id}`)}
                searchPlaceholder="Search videos..."
            />

            <UploadModal
                isOpen={isUploadModalOpen}
                onClose={() => setIsUploadModalOpen(false)}
                onUpload={async (videoFile) => {
                    try {
                        await uploadVideo(videoFile);
                    } catch (error) {
                        console.error("Upload failed:", error);
                        showToast.error("Failed to upload video.");
                    }
                }}
            />
            <EditVideoModal
                isOpen={!!videoToEdit}
                onClose={() => setVideoToEdit(null)}
                video={videoToEdit}
                onUpdate={async (videoId, videoData) => {
                    try {
                        await updateVideo(videoId, videoData);
                    } catch (error) {
                        console.error("Update failed:", error);
                        showToast.error("Failed to update video.");
                    }
                }}
            />
            <DeleteVideoModal
                isOpen={!!videoToDelete}
                onClose={() => setVideoToDelete(null)}
                video={videoToDelete}
                onDelete={async (videoId) => {
                    try {
                        await deleteVideo(videoId);
                    } catch (error) {
                        console.error("Delete failed:", error);
                        showToast.error("Failed to delete video.");
                    }
                }}
            />
        </div>
    );
};

export default Dashboard;
