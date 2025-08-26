// src/pages/Dashboard.jsx

import React, { useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import {
    Upload,
    Play,
    CheckCircle,
    RefreshCw,
    AlertTriangle,
    Trash2,
    Clock,
    Loader2 as ProcessingIcon,
    ShieldX,
    ShieldCheck,
    Activity,
    HelpCircle,
    FileVideo,
    ChartCandlestickIcon,
    Copy,
    Edit,
    ChartNetwork,
    Brain,
    Download,
} from "lucide-react";
import {
    useVideosQuery,
    useVideoStats,
    useUploadVideoMutation,
    useUpdateMediaMutation,
    useDeleteMediaMutation,
} from "../hooks/useMediaQuery.jsx";
import { PageHeader } from "../components/layout/PageHeader.jsx";
import { StatCard } from "../components/ui/StatCard";
import { DataTable } from "../components/ui/DataTable";
import { Button } from "../components/ui/Button";
import { UploadModal } from "../components/media/UploadModal.jsx";
import { EditVideoModal } from "../components/media/EditVideoModal.js";
import { DeleteVideoModal } from "../components/media/DeleteVideoModal.js";
import { VideoSearchFilter } from "../components/media/VideoSearchFilter.jsx";
import ModelSelectionModal from "../components/analysis/ModelSelectionModal.jsx";
import { formatDate } from "../utils/formatters.js";
import { SkeletonCard } from "../components/ui/SkeletonCard.jsx";
import showToast from "../utils/toast.js";
import { DownloadService } from "../services/DownloadReport.js";
import { useAuth } from "../contexts/AuthContext.jsx";
import { useServerStatusQuery } from "../hooks/useMonitoringQuery.js";

// --- SUB-COMPONENTS (Logic Preserved, UI Refined) ---

const DashboardSkeleton = () => (
    <div className="space-y-8">
        <div className="flex justify-between items-center">
            <SkeletonCard className="h-10 w-64" />
            <SkeletonCard className="h-10 w-32" />
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            <SkeletonCard className="h-32" />
            <SkeletonCard className="h-32" />
            <SkeletonCard className="h-32" />
            <SkeletonCard className="h-32" />
        </div>
        <SkeletonCard className="h-20" />
        <SkeletonCard className="h-96" />
    </div>
);

const StatusBadge = ({ status }) => {
    const styles = {
        ANALYZED: "bg-green-500/10 text-green-500",
        PARTIALLY_ANALYZED: "bg-blue-500/10 text-blue-500",
        PROCESSING: "bg-yellow-500/10 text-yellow-500",
        QUEUED: "bg-indigo-500/10 text-indigo-500",
        UPLOADED: "bg-gray-500/10 text-gray-500",
        FAILED: "bg-red-500/10 text-red-500",
    };
    const Icon = null;
    return (
        <div
            className={`inline-flex items-center gap-2 text-xs font-semibold p-2 px-3 rounded-full capitalize ${
                styles[status] || styles["UPLOADED"]
            }`}
        >
            {Icon && (
                <Icon
                    className={`w-4 h-4 ${
                        status === "PROCESSING" ? "animate-spin" : ""
                    }`}
                />
            )}
            <span>{status.replace("_", " ").toLowerCase()}</span>
        </div>
    );
};

const AnalysisSummary = ({ analyses = [] }) => {
    const completed = analyses.filter((a) => a.status === "COMPLETED");
    if (completed.length === 0)
        return (
            <span className="text-sm text-light-muted-text dark:text-dark-muted-text italic">
                Awaiting results...
            </span>
        );
    const fakes = completed.filter((a) => a.prediction === "FAKE").length;
    const reals = completed.length - fakes;
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
        <div className="flex flex-row">
            <div className={`flex items-center font-semibold text-sm ${color}`}>
                <Icon className="w-6 h-6 mr-2 flex-shrink-0" />
                <div className="flex flex-col">
                    {consensus}
                    <span className="text-xs text-light-muted-text dark:text-dark-muted-text mt-1">
                        {completed.length} Models Analyzed
                    </span>
                </div>
            </div>
        </div>
    );
};

// --- MAIN DASHBOARD PAGE ---

export const Dashboard = () => {
    // --- STATE AND LOGIC (PRESERVED) ---
    const {
        data: videos = [],
        isLoading,
        refetch,
        isRefetching,
    } = useVideosQuery();
    const { stats } = useVideoStats();
    const { data: serverStatus } = useServerStatusQuery();
    const totalAvailableModels =
        serverStatus?.modelsInfo?.filter((m) => m.loaded).length || 0;
    const uploadMutation = useUploadVideoMutation();
    const updateMutation = useUpdateMediaMutation();
    const deleteMutation = useDeleteMediaMutation();
    const navigate = useNavigate();
    const [modal, setModal] = useState({ type: null, data: null });
    const [searchTerm, setSearchTerm] = useState("");
    const [statusFilter, setStatusFilter] = useState("ALL");

    // get user's data from authmiddleware
    const { user } = useAuth();

    // All memoized logic is preserved.
    const filteredVideos = useMemo(() => {
        let filtered = [...videos];
        if (searchTerm.trim()) {
            const searchLower = searchTerm.toLowerCase();
            filtered = filtered.filter((v) =>
                v.filename?.toLowerCase().includes(searchLower)
            );
        }
        if (statusFilter !== "ALL") {
            filtered = filtered.filter((v) => v.status === statusFilter);
        }
        filtered.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
        return filtered;
    }, [videos, searchTerm, statusFilter]);

    const [isDownloadingPDF, setIsDownloadingPDF] = useState(false);

    const columns = useMemo(
        () => [
            {
                key: "filename",
                header: "File",
                render: (item) => (
                    <span className="font-semibold">{item.filename}</span>
                ),
            },
            {
                key: "description",
                header: "Description",
                render: (item) =>
                    item.description || "No description provided.",
            },
            {
                key: "status",
                header: "Job Status",
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
                render: (item) => formatDate(item.createdAt),
            },
            {
                key: "actions",
                header: "Actions",
                render: (item) => (
                    <div className="flex items-center justify-start space-x-1">
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
                            title="Edit Video Details"
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
                            title="Download Latest Video Report"
                            onClick={async (e) => {
                                e.stopPropagation();
                                setIsDownloadingPDF(true);
                                try {
                                    await DownloadService.generateAndDownloadPDFPrint(
                                        item,
                                        user
                                    );
                                    showToast.success(
                                        "PDF report generated successfully!",
                                        { duration: 10000 }
                                    );
                                } catch (err) {
                                    console.error(
                                        "PDF generation failed:",
                                        err
                                    );
                                    showToast.error(
                                        err.message ||
                                            "Failed to generate PDF report."
                                    );
                                } finally {
                                    setIsDownloadingPDF(false);
                                }
                            }}
                            disabled={
                                item.status !== "ANALYZED" || isDownloadingPDF
                            }
                        >
                            <Download className="h-5 w-5 text-green-500" />
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
                                setModal({ type: "delete", data: item });
                            }}
                        >
                            <Trash2 className="h-5 w-5 text-red-500" />
                        </Button>
                    </div>
                ),
            },
        ],
        [navigate, user, isDownloadingPDF]
    );
    // --- RENDER LOGIC ---
    if (isLoading && !videos.length) return <DashboardSkeleton />;

    return (
        <div className="space-y-4">
            <PageHeader
                title="Dashboard"
                description="Manage, upload, and review your video analyses."
                actions={
                    <>
                        <Button
                            onClick={() => {
                                refetch();
                                showToast.success(
                                    "Data refreshed successfully!"
                                );
                            }}
                            isLoading={isRefetching}
                            variant="outline"
                        >
                            <RefreshCw className="mr-2 h-4 w-4" /> Refresh
                        </Button>
                        <Button onClick={() => setModal({ type: "upload" })}>
                            <Upload className="mr-2 h-4 w-4" /> Upload Video
                        </Button>
                    </>
                }
            />

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <StatCard
                    title="Total Videos"
                    value={stats.total}
                    icon={Play}
                    isLoading={isLoading}
                    onClick={() => console.log("Total Videos Clicked")}
                    cardColor="blue"
                />
                <StatCard
                    title="Total Analyses"
                    value={stats.totalAnalyses}
                    icon={ChartNetwork}
                    isLoading={isLoading}
                    onClick={() => console.log("Total Analyses Clicked")}
                    cardColor="purple"
                />
                <StatCard
                    title="Authentic Results"
                    value={stats.realDetections}
                    icon={ShieldCheck}
                    isLoading={isLoading}
                    onClick={() => console.log("Authentic Results Clicked")}
                    cardColor="green"
                />
                <StatCard
                    title="Deepfake Results"
                    value={stats.fakeDetections}
                    icon={ShieldX}
                    isLoading={isLoading}
                    onClick={() => console.log("Deepfake Results Clicked")}
                    cardColor="red"
                />
            </div>

            <DataTable
                columns={columns}
                data={filteredVideos}
                loading={isRefetching}
                title="Video Results"
                searchPlaceholder="Search for a video..."
                description={"View and manage your uploaded videos."}
                emptyState={{
                    icon: FileVideo,
                    title: "No Videos Found",
                    message: "Upload one for analysis.",
                    action: (
                        <Button onClick={() => setModal({ type: "upload" })}>
                            <Upload className="mr-2 h-4 w-4" /> Upload Video
                        </Button>
                    ),
                }}
            />

            {/* --- MODALS (Preserved) --- */}
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
                onAnalysisStart={refetch}
            />
        </div>
    );
};

export default Dashboard;
