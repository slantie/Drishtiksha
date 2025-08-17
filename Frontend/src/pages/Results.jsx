// src/pages/Results.jsx

import React, { useState, useContext } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import {
    ArrowLeft,
    Edit,
    Trash2,
    Brain,
    Activity,
    ShieldCheck,
    ShieldAlert,
    Clock,
    AlertTriangle,
    FileText,
    HardDrive,
    Video as VideoIcon,
    FileDown,
    Bot,
    Download,
    ChartNetwork,
    FileSpreadsheetIcon,
} from "lucide-react";
import {
    useVideoQuery,
    useUpdateVideoMutation,
    useDeleteVideoMutation,
} from "../hooks/useVideosQuery.jsx";
import { AuthContext } from "../contexts/AuthContext.jsx";
import { Button } from "../components/ui/Button";
import {
    Card,
    CardContent,
    CardHeader,
    CardTitle,
    CardDescription,
    CardFooter,
} from "../components/ui/Card";
import { PageLoader } from "../components/ui/LoadingSpinner";
import { VideoPlayer } from "../components/videos/VideoPlayer.jsx";
import { AnalysisInProgress } from "../components/videos/AnalysisInProgress.jsx";
import { EditVideoModal } from "../components/videos/EditVideoModal.jsx";
import { DeleteVideoModal } from "../components/videos/DeleteVideoModal.jsx";
import ModelSelectionModal from "../components/analysis/ModelSelectionModal.jsx";
import { DownloadService } from "../services/DownloadReport.js";
import { MODEL_INFO } from "../constants/apiEndpoints.js";
import { formatProcessingTime, formatBytes } from "../utils/formatters.js";
import { showToast } from "../utils/toast.js";
import { PageHeader } from "../components/layout/PageHeader.jsx";
import {
    Alert,
    AlertTitle,
    AlertDescription,
} from "../components/ui/Alert.jsx";
import { EmptyState } from "../components/ui/EmptyState.jsx";

// REFACTOR: The result card is now cleaner and uses Card sub-components for better structure.
const AnalysisResultCard = ({ analysis, videoId }) => {
    const isReal = analysis.prediction === "REAL";
    const confidence = analysis.confidence * 100;
    const modelInfo = MODEL_INFO[analysis.model];

    if (analysis.status === "FAILED") {
        return (
            <Card>
                <CardHeader>
                    <CardTitle>{modelInfo?.label || analysis.model}</CardTitle>
                </CardHeader>
                <CardContent>
                    <Alert variant="destructive">
                        <AlertTriangle className="h-4 w-4" />
                        <AlertTitle>Analysis Failed</AlertTitle>
                        <AlertDescription>
                            {analysis.errorMessage ||
                                "An unknown error occurred during analysis."}
                        </AlertDescription>
                    </Alert>
                </CardContent>
            </Card>
        );
    }

    return (
        <Card
            className={`transition-all hover:shadow-lg ${
                isReal ? "border-green-500/30" : "border-red-500/30"
            }`}
        >
            <CardHeader className="flex flex-row items-start justify-between">
                <div>
                    <CardTitle>{modelInfo?.label || analysis.model}</CardTitle>
                    <CardDescription>{modelInfo?.description}</CardDescription>
                </div>
                {isReal ? (
                    <ShieldCheck className="w-8 h-8 text-green-500" />
                ) : (
                    <ShieldAlert className="w-8 h-8 text-red-500" />
                )}
            </CardHeader>
            <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4 text-center">
                    <div>
                        <p
                            className={`text-3xl font-bold ${
                                isReal ? "text-green-600" : "text-red-600"
                            }`}
                        >
                            {analysis.prediction}
                        </p>
                        <p className="text-xs text-light-muted-text dark:text-dark-muted-text">
                            Prediction
                        </p>
                    </div>
                    <div>
                        <p className="text-3xl font-bold">
                            {confidence.toFixed(1)}%
                        </p>
                        <p className="text-xs text-light-muted-text dark:text-dark-muted-text">
                            Confidence
                        </p>
                    </div>
                </div>
                <div className="text-xs text-light-muted-text dark:text-dark-muted-text pt-2 space-y-1">
                    <div className="flex justify-between">
                        <span>
                            <Clock className="inline w-3 h-3 mr-1" />
                            Processing Time:
                        </span>
                        <span className="font-mono">
                            {formatProcessingTime(analysis.processingTime)}
                        </span>
                    </div>
                    <div className="flex justify-between">
                        <span>
                            <Activity className="inline w-3 h-3 mr-1" />
                            Analysis Type:
                        </span>
                        <span className="font-mono">
                            {analysis.analysisType}
                        </span>
                    </div>
                </div>
            </CardContent>
            <CardFooter>
                <Button asChild variant="outline" className="w-full">
                    <Link to={`/results/${videoId}/${analysis.id}`} className="flex items-center justify-center gap-2">
                        <ChartNetwork className="h-5 w-5 text-purple-500" />
                        View Detailed Report
                    </Link>
                </Button>
            </CardFooter>
        </Card>
    );
};

const Results = () => {
    // --- STATE AND LOGIC (PRESERVED) ---
    const { videoId } = useParams();
    const navigate = useNavigate();
    const { user } = useContext(AuthContext);
    const {
        data: video,
        isLoading,
        error,
        refetch,
    } = useVideoQuery(videoId, {
        refetchInterval: (query) =>
            ["QUEUED", "PROCESSING"].includes(query.state.data?.status)
                ? 3000
                : false,
    });
    const updateMutation = useUpdateVideoMutation();
    const deleteMutation = useDeleteVideoMutation();
    const [modal, setModal] = useState({ type: null, data: null });
    const [isDownloadingPDF, setIsDownloadingPDF] = useState(false);
    const [isDownloadingVideo, setIsDownloadingVideo] = useState(false);

    const handleDownloadPDF = async () => {
        if (!video) return;
        setIsDownloadingPDF(true);
        try {
            await DownloadService.generateAndDownloadPDF(video, user);
        } catch (err) {
            showToast.error(err.message || "Failed to generate PDF report.");
        } finally {
            setIsDownloadingPDF(false);
        }
    };

    // --- RENDER LOGIC ---
    if (isLoading) return <PageLoader text="Loading Analysis Report..." />;

    if (error)
        return (
            <div className="text-center py-16">
                <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
                <h2 className="text-2xl font-bold">Error Loading Video</h2>
                <p className="mb-6 text-light-muted-text dark:text-dark-muted-text">
                    {error.message}
                </p>
                <Button asChild>
                    <Link to="/dashboard">
                        <ArrowLeft className="mr-2 h-4 w-4" />
                        Back to Dashboard
                    </Link>
                </Button>
            </div>
        );
    if (!video) return <div>Video not found.</div>;
    if (["QUEUED", "PROCESSING"].includes(video.status))
        return <AnalysisInProgress video={video} />;

    const completedAnalyses =
        video.analyses?.filter((a) => a.status === "COMPLETED") || [];
    const failedAnalyses =
        video.analyses?.filter((a) => a.status === "FAILED") || [];
    const allAnalyses = [...completedAnalyses, ...failedAnalyses].sort(
        (a, b) => new Date(b.createdAt) - new Date(a.createdAt)
    );

    return (
        <div className="space-y-4">
            <PageHeader
                title="Analysis Results"
                description={`Detailed report for ${video.filename}`}
                actions={
                    <Button asChild variant="outline">
                        <Link to="/dashboard">
                            <ArrowLeft className="mr-2 h-4 w-4" /> Back to
                            Dashboard
                        </Link>
                    </Button>
                }
            />

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 items-start">
                {/* Left Column: Video Details & Actions */}
                <div className="lg:col-span-1 space-y-4 sticky top-24">
                    <Card>
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                                <VideoIcon className="w-5 h-5 text-primary-main" />
                                Video Details & Actions
                            </CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <VideoPlayer videoUrl={video.url} />
                            <div className="flex items-center justify-between">
                                <div>
                                    <h4 className="font-semibold text-lg">
                                        {video.filename}
                                    </h4>
                                    <p className="text-sm text-light-muted-text dark:text-dark-muted-text mt-1">
                                        {video.description ||
                                            "No description provided."}
                                    </p>
                                </div>

                                <Button
                                    onClick={() =>
                                        DownloadService.downloadVideo(
                                            video.url,
                                            video.filename
                                        )
                                            // Set the state true,
                                            // Set the loading state
                                            .then(() => {
                                                setIsDownloadingVideo(false);
                                                showToast.success(
                                                    "Video downloaded successfully!"
                                                );
                                            })
                                            .catch((error) => {
                                                showToast.error(
                                                    `Failed to download video: ${error.message}`
                                                );
                                                setIsDownloadingVideo(false);
                                            })
                                    }
                                    variant="outline"
                                    isLoading={isDownloadingVideo}
                                >
                                    <Download className="mr-2 h-4 w-4" />{" "}
                                    {isDownloadingVideo
                                        ? "Downloading..."
                                        : "Download Video"}
                                </Button>
                            </div>
                            <div className="text-sm space-y-1 pt-2">
                                <div className="flex justify-between">
                                    <span>
                                        <FileText className="inline w-3 h-3 mr-1" />
                                        MIME Type:
                                    </span>
                                    <span className="font-mono">
                                        {video.mimetype}
                                    </span>
                                </div>
                                <div className="flex justify-between">
                                    <span>
                                        <HardDrive className="inline w-3 h-3 mr-1" />
                                        File Size:
                                    </span>
                                    <span className="font-mono">
                                        {formatBytes(video.size)}
                                    </span>
                                </div>
                            </div>
                        </CardContent>
                        <CardContent className="grid grid-cols-2 gap-2">
                            {/* REFACTOR: Button text and icons are now color-coded to match the Dashboard actions for consistency. */}
                            <Button
                                onClick={() =>
                                    setModal({
                                        type: "new_analysis",
                                        data: video,
                                    })
                                }
                                variant="outline"
                                title="Run a New Analysis"
                            >
                                <Brain className="h-4 w-4 mr-2 text-cyan-500" />{" "}
                                Re-Analyze Video
                            </Button>

                            <Button
                                onClick={() =>
                                    setModal({ type: "edit", data: video })
                                }
                                variant="outline"
                                title="Edit Video Details"
                            >
                                <Edit className="h-4 w-4 mr-2 text-blue-500" />{" "}
                                Edit Video Details
                            </Button>

                            <Button
                                onClick={handleDownloadPDF}
                                variant="outline"
                                isLoading={isDownloadingPDF}
                                title="Download PDF Report"
                            >
                                <FileSpreadsheetIcon className="h-4 w-4 mr-2 text-green-500" />{" "}
                                Download PDF Report
                            </Button>

                            <Button
                                onClick={() =>
                                    setModal({ type: "delete", data: video })
                                }
                                variant="destructive"
                                title="Delete Video"
                            >
                                <Trash2 className="h-4 w-4 mr-2" />
                                Delete Video
                            </Button>
                        </CardContent>
                    </Card>
                </div>

                {/* Right Column: Analysis Results */}
                <div className="lg:col-span-2 space-y-4">
                    {allAnalyses.length === 0 ? (
                        <EmptyState
                            icon={Bot}
                            title="No Analysis Results"
                            message="This video is awaiting its first analysis. Run one to get started."
                            action={
                                <Button
                                    onClick={() =>
                                        setModal({
                                            type: "new_analysis",
                                            data: video,
                                        })
                                    }
                                >
                                    <Brain className="mr-2 h-4 w-4" /> Start
                                    First Analysis
                                </Button>
                            }
                        />
                    ) : (
                        allAnalyses.map((analysis) => (
                            <AnalysisResultCard
                                key={analysis.id}
                                analysis={analysis}
                                videoId={video.id}
                            />
                        ))
                    )}
                </div>
            </div>

            {/* Modals */}
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
                onDelete={(videoId) =>
                    deleteMutation
                        .mutateAsync(videoId)
                        .then(() => navigate("/dashboard"))
                }
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

export default Results;
