// src/pages/Dashboard.jsx

import React, { useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import {
    Upload, Play, ShieldX, ShieldCheck, HelpCircle, FileVideo, Copy, Edit, ChartNetwork, Download, Trash2, FileAudio, FileImage
} from "lucide-react";
// UPDATED: Importing the new generic hooks
import {
    useMediaQuery,
    useMediaStats,
    useUploadMediaMutation,
    useUpdateMediaMutation,
    useDeleteMediaMutation,
} from "../hooks/useMediaQuery.jsx";
import { PageHeader } from "../components/layout/PageHeader.jsx";
import { StatCard } from "../components/ui/StatCard";
import { DataTable } from "../components/ui/DataTable";
import { Button } from "../components/ui/Button";
// UPDATED: Importing the new generic modal components from the 'media' directory
import { UploadModal } from "../components/media/UploadModal.jsx";
import { EditMediaModal } from "../components/media/EditMediaModal.jsx";
import { DeleteMediaModal } from "../components/media/DeleteMediaModal.jsx";
// Note: VideoSearchFilter will be renamed later, but its functionality is already generic.
import { VideoSearchFilter } from "../components/media/VideoSearchFilter.jsx";
import ModelSelectionModal from "../components/analysis/ModelSelectionModal.jsx";
import { formatDate } from "../utils/formatters.js";
import { SkeletonCard } from "../components/ui/SkeletonCard.jsx";
import showToast from "../utils/toast.js";
import { DownloadService } from "../services/DownloadReport.js";
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
        <div className={`inline-flex items-center gap-2 text-xs font-semibold px-3 py-1 rounded-full capitalize ${styles[status] || "bg-gray-500/10 text-gray-500"}`}>
            <span>{status.replace("_", " ").toLowerCase()}</span>
        </div>
    );
};

const MediaTypeIcon = ({ mediaType }) => {
    switch (mediaType) {
        case 'VIDEO': return <FileVideo className="w-5 h-5 text-purple-500" />;
        case 'AUDIO': return <FileAudio className="w-5 h-5 text-blue-500" />;
        case 'IMAGE': return <FileImage className="w-5 h-5 text-green-500" />;
        default: return <FileVideo className="w-5 h-5 text-gray-500" />;
    }
};

const AnalysisSummary = ({ analyses = [] }) => {
    const completed = analyses.filter((a) => a.status === "COMPLETED");
    if (completed.length === 0) return <span className="text-sm text-light-muted-text dark:text-dark-muted-text italic">Awaiting results...</span>;
    
    const fakes = completed.filter((a) => a.prediction === "FAKE").length;
    const reals = completed.length - fakes;
    
    let consensus, Icon, color;
    if (fakes > reals) { consensus = "Deepfake Detected"; Icon = ShieldX; color = "text-red-500"; }
    else if (reals > fakes) { consensus = "Authentic"; Icon = ShieldCheck; color = "text-green-500"; }
    else { consensus = "Inconclusive"; Icon = HelpCircle; color = "text-yellow-500"; }

    return (
        <div className={`flex items-center font-semibold text-sm ${color}`}>
            <Icon className="w-5 h-5 mr-2 flex-shrink-0" />
            <div>
                {consensus}
                <span className="block text-xs font-normal text-light-muted-text dark:text-dark-muted-text">{completed.length} Model(s) Analyzed</span>
            </div>
        </div>
    );
};

// --- MAIN DASHBOARD PAGE ---

export const Dashboard = () => {
    // UPDATED: Using new generic hooks
    const { data: mediaItems = [], isLoading, refetch, isRefetching } = useMediaQuery();
    const { stats } = useMediaStats();
    const uploadMutation = useUploadMediaMutation();
    const updateMutation = useUpdateMediaMutation();
    const deleteMutation = useDeleteMediaMutation();
    
    const navigate = useNavigate();
    const { user } = useAuth();
    const [modal, setModal] = useState({ type: null, data: null });
    const [searchTerm, setSearchTerm] = useState("");
    const [statusFilter, setStatusFilter] = useState("ALL");

    // RENAMED: from filteredVideos to filteredMedia
    const filteredMedia = useMemo(() => {
        let filtered = [...mediaItems];
        if (searchTerm.trim()) {
            const searchLower = searchTerm.toLowerCase();
            filtered = filtered.filter(v => v.filename?.toLowerCase().includes(searchLower));
        }
        if (statusFilter !== "ALL") {
            filtered = filtered.filter(v => v.status === statusFilter);
        }
        return filtered.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
    }, [mediaItems, searchTerm, statusFilter]);

    const [isDownloadingPDF, setIsDownloadingPDF] = useState(false);

    const columns = useMemo(() => [
        // NEW: mediaType column
        { key: "mediaType", header: "Type", render: (item) => <MediaTypeIcon mediaType={item.mediaType} /> },
        { key: "filename", header: "File", render: (item) => <span className="font-semibold">{item.filename}</span> },
        { key: "status", header: "Job Status", render: (item) => <StatusBadge status={item.status} /> },
        { key: "summary", header: "Analysis Result", render: (item) => <AnalysisSummary analyses={item.analyses} /> },
        { key: "createdAt", header: "Uploaded On", render: (item) => formatDate(item.createdAt) },
        { key: "actions", header: "Actions", render: (item) => (
            <div className="flex items-center justify-end space-x-1">
                <Button variant="ghost" size="icon" title="View Results" onClick={(e) => { e.stopPropagation(); navigate(`/results/${item.id}`); }}>
                    <ChartNetwork className="h-5 w-5 text-purple-500" />
                </Button>
                <Button variant="ghost" size="icon" title="Edit Details" onClick={(e) => { e.stopPropagation(); setModal({ type: "edit", data: item }); }}>
                    <Edit className="h-5 w-5 text-blue-500" />
                </Button>
                <Button variant="ghost" size="icon" title="Download Report" onClick={async (e) => {
                    e.stopPropagation();
                    setIsDownloadingPDF(true);
                    try {
                        await DownloadService.generateAndDownloadPDF(item, user);
                    } catch (err) {
                        showToast.error(err.message || "Failed to generate PDF report.");
                    } finally {
                        setIsDownloadingPDF(false);
                    }
                }} disabled={item.status !== "ANALYZED" || isDownloadingPDF}>
                    <Download className="h-5 w-5 text-green-500" />
                </Button>
                <Button variant="ghost" size="icon" title="Delete Media" onClick={(e) => { e.stopPropagation(); setModal({ type: "delete", data: item }); }}>
                    <Trash2 className="h-5 w-5 text-red-500" />
                </Button>
            </div>
        )},
    ], [navigate, user, isDownloadingPDF]);

    if (isLoading && !mediaItems.length) return <DashboardSkeleton />;

    return (
        <div className="space-y-4">
            <PageHeader
                title="Dashboard"
                description="Manage, upload, and review your media analyses."
                actions={
                    <Button onClick={() => setModal({ type: "upload" })}>
                        <Upload className="mr-2 h-4 w-4" /> Upload Media
                    </Button>
                }
            />

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <StatCard title="Total Files" value={stats.total} icon={Play} isLoading={isLoading} cardColor="blue" />
                <StatCard title="Total Analyses" value={stats.totalAnalyses} icon={ChartNetwork} isLoading={isLoading} cardColor="purple" />
                <StatCard title="Authentic Results" value={stats.realDetections} icon={ShieldCheck} isLoading={isLoading} cardColor="green" />
                <StatCard title="Deepfake Results" value={stats.fakeDetections} icon={ShieldX} isLoading={isLoading} cardColor="red" />
            </div>

            {/* UPDATED: Renamed props */}
            <VideoSearchFilter
                searchTerm={searchTerm}
                statusFilter={statusFilter}
                onSearchChange={setSearchTerm}
                onStatusFilterChange={setStatusFilter}
                videos={mediaItems} // Prop name remains for now, but data is generic
            />

            <DataTable
                columns={columns}
                // UPDATED: Passing generic media data
                data={filteredMedia}
                loading={isRefetching}
                title="Media Files"
                description="View and manage your uploaded media."
                searchPlaceholder="Search by filename..."
                onRowClick={(item) => navigate(`/results/${item.id}`)}
                emptyState={{
                    icon: FileVideo,
                    title: "No Media Found",
                    message: "Upload a file to begin your first analysis.",
                    action: <Button onClick={() => setModal({ type: "upload" })}>
                                <Upload className="mr-2 h-4 w-4" /> Upload Media
                            </Button>
                }}
            />

            {/* --- MODALS (Updated with generic props) --- */}
            <UploadModal
                isOpen={modal.type === "upload"}
                onClose={() => setModal({ type: null })}
            />
            <EditMediaModal
                isOpen={modal.type === "edit"}
                onClose={() => setModal({ type: null })}
                media={modal.data}
                onUpdate={(mediaId, data) => updateMutation.mutateAsync({ mediaId, updateData: data })}
            />
            <DeleteMediaModal
                isOpen={modal.type === "delete"}
                onClose={() => setModal({ type: null })}
                media={modal.data}
                onDelete={deleteMutation.mutateAsync}
            />
            <ModelSelectionModal
                isOpen={modal.type === "new_analysis"}
                onClose={() => setModal({ type: null })}
                // Note: Prop name is videoId but it's a generic ID
                videoId={modal.data?.id}
                onAnalysisStart={refetch}
            />
        </div>
    );
};

export default Dashboard;