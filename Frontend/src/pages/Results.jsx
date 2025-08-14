// src/pages/Results.jsx

import React, { useState, useContext } from "react"; // Import useContext
import { useParams, Link } from "react-router-dom";
import {
    Loader2,
    AlertTriangle,
    ArrowLeft,
    Cpu,
    ShieldCheck,
    ShieldAlert,
    RefreshCw,
    Edit,
    Trash,
    Calendar,
    FileText,
    HardDrive,
    Clock,
    Activity,
    TrendingUp,
    Video as VideoIcon,
    Download,
    FileDown,
} from "lucide-react";
import {
    useVideoQuery,
    useUpdateVideoMutation,
    useDeleteVideoMutation,
} from "../hooks/useVideosQuery.js";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { PageLoader } from "../components/ui/LoadingSpinner";
import { EditVideoModal } from "../components/videos/EditVideoModal";
import { DeleteVideoModal } from "../components/videos/DeleteVideoModal";
import { VideoPlayer } from "../components/videos/VideoPlayer.jsx";
import { DownloadService } from "../services/DownloadReport.js";
import { AuthContext } from "../contexts/AuthContext.jsx";
import showToast from "../utils/toast.js";

// Helper functions remain the same...
const formatBytes = (bytes) =>
    bytes ? `${(bytes / 1024 / 1024).toFixed(2)} MB` : "N/A";
const formatDate = (dateString) =>
    new Date(dateString).toLocaleString("en-US", {
        year: "numeric",
        month: "long",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
    });
const formatProcessingTime = (timeInSeconds) => {
    if (!timeInSeconds) return "N/A";
    if (timeInSeconds < 60) return `${timeInSeconds.toFixed(1)}s`;
    const minutes = Math.floor(timeInSeconds / 60);
    const seconds = (timeInSeconds % 60).toFixed(1);
    return `${minutes}m ${seconds}s`;
};
const getStatusColor = (status) => {
    const statusColors = {
        UPLOADED: "bg-blue-500/10 text-blue-600 border-blue-200",
        PROCESSING: "bg-yellow-500/10 text-yellow-600 border-yellow-200",
        ANALYZED: "bg-green-500/10 text-green-600 border-green-200",
        FAILED: "bg-red-500/10 text-red-600 border-red-200",
    };
    return (
        statusColors[status] || "bg-gray-500/10 text-gray-600 border-gray-200"
    );
};
const getAnalysisStatusColor = (status) => {
    const statusColors = {
        PENDING: "bg-gray-500/10 text-gray-600 border-gray-200",
        PROCESSING: "bg-yellow-500/10 text-yellow-600 border-yellow-200",
        COMPLETED: "bg-green-500/10 text-green-600 border-green-200",
        FAILED: "bg-red-500/10 text-red-600 border-red-200",
    };
    return (
        statusColors[status] || "bg-gray-500/10 text-gray-600 border-gray-200"
    );
};

// UI Components (ResultsHeader, AnalysisCard, etc.) remain unchanged...
const ResultsHeader = ({ video, onRefresh, onEditClick, onDeleteClick }) => (
    <Card>
        <div className="flex flex-col space-y-4">
            {/* Main Header */}
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                    <Link
                        to="/dashboard"
                        className="text-blue-600 hover:text-blue-800 mr-2 bg-blue-300/30 dark:bg-blue-800/30 rounded-full p-2"
                    >
                        <ArrowLeft className="h-6 w-6" />
                    </Link>
                    <div>
                        <h1 className="text-3xl font-bold">Analysis Results</h1>
                        <p className="text-light-muted-text dark:text-dark-muted-text mt-1">
                            File:{" "}
                            <span className="text-primary-main font-medium">
                                {video.filename}
                            </span>
                        </p>
                    </div>
                </div>
                <div className="flex items-center gap-2 flex-wrap">
                    <Button
                        onClick={onRefresh}
                        variant="outline"
                        className="py-3 px-4"
                    >
                        <RefreshCw className="mr-2 h-5 w-5" /> Refresh
                    </Button>
                    <Button
                        onClick={onEditClick}
                        variant="outline"
                        className="py-3 px-4"
                    >
                        <Edit className="h-5 w-5 mr-2" /> Edit
                    </Button>
                    <Button
                        onClick={onDeleteClick}
                        variant="destructive"
                        className="py-3 px-4"
                    >
                        <Trash className="h-5 w-5 mr-2" /> Delete
                    </Button>
                </div>
            </div>

            {/* Status and Quick Info */}
            <div className="flex flex-wrap items-center gap-4 pt-2 border-t border-gray-200 dark:border-gray-700">
                <div className="flex items-center gap-2">
                    <VideoIcon className="h-4 w-4 text-gray-500" />
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                        Status:
                    </span>
                    <span
                        className={`px-2 py-1 rounded-full text-xs font-semibold border ${getStatusColor(
                            video.status
                        )}`}
                    >
                        {video.status}
                    </span>
                </div>
                <div className="flex items-center gap-2">
                    <HardDrive className="h-4 w-4 text-gray-500" />
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                        Size:
                    </span>
                    <span className="text-sm font-medium">
                        {formatBytes(video.size)}
                    </span>
                </div>
                <div className="flex items-center gap-2">
                    <Calendar className="h-4 w-4 text-gray-500" />
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                        Uploaded:
                    </span>
                    <span className="text-sm font-medium">
                        {formatDate(video.createdAt)}
                    </span>
                </div>
                {video.analyses?.length > 0 && (
                    <div className="flex items-center gap-2">
                        <Activity className="h-4 w-4 text-gray-500" />
                        <span className="text-sm text-gray-600 dark:text-gray-400">
                            Analyses:
                        </span>
                        <span className="text-sm font-medium">
                            {video.analyses.length} completed
                        </span>
                    </div>
                )}
            </div>
        </div>
    </Card>
);

const AnalysisCard = ({ analysis }) => {
    const isReal = analysis.prediction === "REAL";
    const confidence = analysis.confidence * 100;

    return (
        <Card
            className={`border-2 ${
                isReal ? "border-green-500/30" : "border-red-500/30"
            }`}
        >
            {/* Header with model name and prediction icon */}
            <div className="flex justify-between items-start mb-6">
                <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800">
                        <Cpu className="h-5 w-5 text-gray-600 dark:text-gray-400" />
                    </div>
                    <div>
                        <h3 className="text-xl font-bold">{analysis.model}</h3>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                            Analysis Model
                        </p>
                    </div>
                </div>
                <div
                    className={`p-2 rounded-lg ${
                        isReal
                            ? "bg-green-100 dark:bg-green-900"
                            : "bg-red-100 dark:bg-red-900"
                    }`}
                >
                    {isReal ? (
                        <ShieldCheck className={`h-7 w-7 text-green-600`} />
                    ) : (
                        <ShieldAlert className={`h-7 w-7 text-red-600`} />
                    )}
                </div>
            </div>

            {/* Main confidence display */}
            <div className="text-center mb-6">
                <p
                    className={`text-5xl font-bold mb-2 ${
                        isReal ? "text-green-600" : "text-red-600"
                    }`}
                >
                    {confidence.toFixed(1)}%
                </p>
                <p className="text-lg font-semibold mb-1">
                    {isReal ? "Likely Authentic" : "Likely Deepfake"}
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                    Confidence Level
                </p>
            </div>

            {/* Analysis details */}
            <div className="space-y-3 pt-4 border-t border-gray-200 dark:border-gray-700">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <Activity className="h-4 w-4 text-gray-500" />
                        <span className="text-sm text-gray-600 dark:text-gray-400">
                            Status:
                        </span>
                    </div>
                    <span
                        className={`px-2 py-1 rounded-full text-xs font-semibold border ${getAnalysisStatusColor(
                            analysis.status
                        )}`}
                    >
                        {analysis.status}
                    </span>
                </div>

                {analysis.processingTime && (
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                            <Clock className="h-4 w-4 text-gray-500" />
                            <span className="text-sm text-gray-600 dark:text-gray-400">
                                Processing Time:
                            </span>
                        </div>
                        <span className="text-sm font-medium">
                            {formatProcessingTime(analysis.processingTime)}
                        </span>
                    </div>
                )}

                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <Calendar className="h-4 w-4 text-gray-500" />
                        <span className="text-sm text-gray-600 dark:text-gray-400">
                            Analyzed:
                        </span>
                    </div>
                    <span className="text-sm font-medium">
                        {formatDate(analysis.createdAt)}
                    </span>
                </div>

                {analysis.errorMessage && (
                    <div className="mt-3 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                        <div className="flex items-center gap-2">
                            <AlertTriangle className="h-4 w-4 text-red-600" />
                            <span className="text-sm font-medium text-red-800 dark:text-red-200">
                                Error:
                            </span>
                        </div>
                        <p className="text-sm text-red-700 dark:text-red-300 mt-1">
                            {analysis.errorMessage}
                        </p>
                    </div>
                )}
            </div>
        </Card>
    );
};

const VideoDetailsCard = ({
    video,
    onDownloadVideo,
    onDownloadPDF,
    // onDownloadHTML,
    isDownloadingVideo,
    isDownloadingPDF,
    // isDownloadingHTML,
}) => (
    <Card>
        {/* Video Player */}
        <div className="aspect-video rounded-lg overflow-hidden bg-black mb-6">
            <VideoPlayer videoUrl={video.url} />
        </div>

        {/* Download Buttons */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-6">
            <Button
                onClick={() => onDownloadVideo(video.url, video.filename)}
                className="w-full"
                variant="outline"
                disabled={isDownloadingVideo}
            >
                {isDownloadingVideo ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                    <Download className="mr-2 h-4 w-4" />
                )}
                {isDownloadingVideo ? "Downloading..." : "Download Video"}
            </Button>
            <Button
                onClick={() => onDownloadPDF(video)}
                className="w-full px-2"
                disabled={isDownloadingPDF}
                variant="outline"
            >
                {isDownloadingPDF ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                    <FileDown className="mr-2 h-4 w-4" />
                )}
                {isDownloadingPDF ? "Generating PDF..." : "PDF Report"}
            </Button>
            {/* <Button
                onClick={() => onDownloadHTML(video)}
                className="w-full"
                variant="outline"
                disabled={isDownloadingHTML}
            >
                {isDownloadingHTML ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                    <FileText className="mr-2 h-4 w-4" />
                )}
                {isDownloadingHTML ? "Generating..." : "HTML Report"}
            </Button> */}
        </div>

        {/* Video Information */}
        <div className="space-y-4">
            <h3 className="text-lg font-bold flex items-center gap-2">
                <VideoIcon className="h-5 w-5" />
                Video Information
            </h3>

            <div className="grid grid-cols-1 gap-3">
                <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                    <div className="flex items-center gap-2">
                        <FileText className="h-4 w-4 text-gray-500" />
                        <span className="text-sm font-medium">Filename:</span>
                    </div>
                    <span className="text-sm text-gray-600 dark:text-gray-400 font-mono break-all">
                        {video.filename}
                    </span>
                </div>

                <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                    <div className="flex items-center gap-2">
                        <HardDrive className="h-4 w-4 text-gray-500" />
                        <span className="text-sm font-medium">File Size:</span>
                    </div>
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                        {formatBytes(video.size)}
                    </span>
                </div>

                <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                    <div className="flex items-center gap-2">
                        <VideoIcon className="h-4 w-4 text-gray-500" />
                        <span className="text-sm font-medium">Format:</span>
                    </div>
                    <span className="text-sm text-gray-600 dark:text-gray-400 uppercase">
                        {video.mimetype?.split("/")[1] || "Unknown"}
                    </span>
                </div>

                <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                    <div className="flex items-center gap-2">
                        <Calendar className="h-4 w-4 text-gray-500" />
                        <span className="text-sm font-medium">Uploaded:</span>
                    </div>
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                        {formatDate(video.createdAt)}
                    </span>
                </div>

                {video.updatedAt !== video.createdAt && (
                    <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                        <div className="flex items-center gap-2">
                            <Edit className="h-4 w-4 text-gray-500" />
                            <span className="text-sm font-medium">
                                Last Modified:
                            </span>
                        </div>
                        <span className="text-sm text-gray-600 dark:text-gray-400">
                            {formatDate(video.updatedAt)}
                        </span>
                    </div>
                )}
            </div>

            {/* Description */}
            {video.description && (
                <div className="mt-4">
                    <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
                        <FileText className="h-4 w-4" />
                        Description
                    </h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-800 p-3 rounded-lg">
                        {video.description}
                    </p>
                </div>
            )}
        </div>
    </Card>
);

const AnalysisSummary = ({ analyses }) => {
    if (!analyses || analyses.length === 0) return null;

    const completedAnalyses = analyses.filter((a) => a.status === "COMPLETED");
    const realCount = completedAnalyses.filter(
        (a) => a.prediction === "REAL"
    ).length;
    const fakeCount = completedAnalyses.filter(
        (a) => a.prediction === "FAKE"
    ).length;
    const avgConfidence =
        completedAnalyses.length > 0
            ? (completedAnalyses.reduce((sum, a) => sum + a.confidence, 0) /
                  completedAnalyses.length) *
              100
            : 0;

    return (
        <Card className="mb-6">
            <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                Analysis Summary
            </h3>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="text-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                    <p className="text-2xl font-bold text-blue-600">
                        {completedAnalyses.length}
                    </p>
                    <p className="text-sm text-blue-600/80">Completed</p>
                </div>

                <div className="text-center p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                    <p className="text-2xl font-bold text-green-600">
                        {realCount}
                    </p>
                    <p className="text-sm text-green-600/80">Real Detections</p>
                </div>

                <div className="text-center p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
                    <p className="text-2xl font-bold text-red-600">
                        {fakeCount}
                    </p>
                    <p className="text-sm text-red-600/80">Fake Detections</p>
                </div>
            </div>

            {completedAnalyses.length > 0 && (
                <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                    <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">
                            Average Confidence:
                        </span>
                        <span className="text-lg font-bold">
                            {avgConfidence.toFixed(1)}%
                        </span>
                    </div>
                </div>
            )}
        </Card>
    );
};

// Main Results Component
export const Results = () => {
    const { videoId } = useParams();

    // TanStack Query hooks
    const {
        data: video,
        isLoading,
        error,
        refetch: fetchVideo,
    } = useVideoQuery(videoId);
    const updateMutation = useUpdateVideoMutation();
    const deleteMutation = useDeleteVideoMutation();

    const { user } = useContext(AuthContext); // Get user from context

    const [isEditModalOpen, setIsEditModalOpen] = useState(false);
    const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
    const [isDownloadingVideo, setIsDownloadingVideo] = useState(false);
    const [isDownloadingPDF, setIsDownloadingPDF] = useState(false);
    const [isDownloadingHTML, setIsDownloadingHTML] = useState(false);

    const handleDownloadVideo = async (videoUrl, filename) => {
        setIsDownloadingVideo(true);
        try {
            await DownloadService.downloadVideo(videoUrl, filename);
        } catch (error) {
            alert(error.message || "Failed to download video");
        } finally {
            setIsDownloadingVideo(false);
        }
    };

    const handleDownloadPDF = async (video) => {
        setIsDownloadingPDF(true);
        try {
            await DownloadService.generateAndDownloadPDF(video, user);
        } catch (error) {
            alert(error.message || "Failed to generate PDF report");
        } finally {
            setIsDownloadingPDF(false);
        }
    };

    const handleDownloadHTML = async (video) => {
        setIsDownloadingHTML(true);
        try {
            // Pass the user object to the service
            await DownloadService.downloadHTMLReport(video, user);
        } catch (error) {
            alert(error.message || "Failed to generate HTML report");
        } finally {
            setIsDownloadingHTML(false);
        }
    };

    // Check for any loading state
    const isAnyLoading =
        isLoading || updateMutation.isPending || deleteMutation.isPending;

    if (isAnyLoading && !video) {
        return <PageLoader text="Loading video analysis..." />;
    }

    if (error) {
        return (
            <div className="text-center p-8 max-w-md mx-auto">
                <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
                <h2 className="text-2xl font-bold mb-2">Error Loading Video</h2>
                <p className="text-light-muted-text dark:text-dark-muted-text mb-6">
                    {error}
                </p>
                <Link to="/dashboard">
                    <Button>
                        <ArrowLeft className="mr-2 h-4 w-4" /> Back to Dashboard
                    </Button>
                </Link>
            </div>
        );
    }

    if (!video) return null;

    return (
        <div className="space-y-6 mx-auto">
            {/* Header */}
            <ResultsHeader
                video={video}
                onRefresh={() => {
                    fetchVideo();
                    showToast.success("Data refreshed!");
                }}
                onEditClick={() => setIsEditModalOpen(true)}
                onDeleteClick={() => setIsDeleteModalOpen(true)}
            />

            {/* Analysis Summary */}
            <AnalysisSummary analyses={video.analyses} />

            {/* Main Content Grid */}
            <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
                <div className="xl:col-span-1">
                    <VideoDetailsCard
                        video={video}
                        onDownloadVideo={handleDownloadVideo}
                        onDownloadPDF={handleDownloadPDF}
                        onDownloadHTML={handleDownloadHTML}
                        isDownloadingVideo={isDownloadingVideo}
                        isDownloadingPDF={isDownloadingPDF}
                        isDownloadingHTML={isDownloadingHTML}
                    />
                </div>

                <div className="xl:col-span-2 space-y-6">
                    {video.analyses?.length > 0 ? (
                        <>
                            <h2 className="text-xl font-bold flex items-center gap-2">
                                <Activity className="h-5 w-5" />
                                Analysis Results
                            </h2>
                            {video.analyses.map((analysis) => (
                                <AnalysisCard
                                    key={analysis.id}
                                    analysis={analysis}
                                />
                            ))}
                        </>
                    ) : (
                        <Card className="text-center p-12">
                            <div className="flex flex-col items-center gap-4">
                                <div className="p-4 bg-gray-100 dark:bg-gray-800 rounded-full">
                                    <Activity className="h-8 w-8 text-gray-400" />
                                </div>
                                <div>
                                    <h2 className="text-xl font-bold mb-2">
                                        No Analyses Available
                                    </h2>
                                    <p className="text-light-muted-text dark:text-dark-muted-text mb-4">
                                        This video is still being processed or
                                        analysis has not started yet.
                                    </p>
                                    <Button
                                        onClick={() => {
                                            fetchVideo();
                                            showToast.success(
                                                "Data refreshed!"
                                            );
                                        }}
                                        variant="outline"
                                        className="py-3 px-4"
                                    >
                                        <RefreshCw className="mr-2 h-5 w-5" />{" "}
                                        Refresh
                                    </Button>
                                </div>
                            </div>
                        </Card>
                    )}
                </div>
            </div>

            {/* Modals */}
            <EditVideoModal
                isOpen={isEditModalOpen}
                onClose={() => setIsEditModalOpen(false)}
                video={video}
                onUpdate={async (videoId, data) => {
                    try {
                        await updateMutation.mutateAsync({
                            videoId,
                            updateData: data,
                        });
                        setIsEditModalOpen(false);
                    } catch (error) {
                        console.error("Update failed:", error);
                        // Toast handled by the mutation
                    }
                }}
            />
            <DeleteVideoModal
                isOpen={isDeleteModalOpen}
                onClose={() => setIsDeleteModalOpen(false)}
                video={video}
                onDelete={async (videoId) => {
                    try {
                        await deleteMutation.mutateAsync(videoId);
                        // Navigation is handled by the mutation
                    } catch (error) {
                        console.error("Delete failed:", error);
                        // Toast handled by the mutation
                    }
                }}
            />
        </div>
    );
};

export default Results;
