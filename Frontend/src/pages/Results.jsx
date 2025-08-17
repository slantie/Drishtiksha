// src/pages/Results.jsx

import React, { useState, useContext } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import {
    ArrowLeft,
    Edit,
    Trash2,
    Download,
    FileDown,
    Brain,
    Activity,
    ShieldCheck,
    ShieldAlert,
    Clock,
    AlertTriangle,
    FileText,
    HardDrive,
    Video as VideoIcon,
} from "lucide-react";
import {
    useVideoQuery,
    useUpdateVideoMutation,
    useDeleteVideoMutation,
} from "../hooks/useVideosQuery.jsx";
import { AuthContext } from "../contexts/AuthContext.jsx";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { PageLoader } from "../components/ui/LoadingSpinner";
import { VideoPlayer } from "../components/videos/VideoPlayer.jsx";
import AnalysisInProgress from "../components/videos/AnalysisInProgress.jsx";
import { EditVideoModal } from "../components/videos/EditVideoModal.jsx";
import { DeleteVideoModal } from "../components/videos/DeleteVideoModal.jsx";
import ModelSelectionModal from "../components/analysis/ModelSelectionModal.jsx";
import { DownloadService } from "../services/DownloadReport.js";
import { MODEL_INFO } from "../constants/apiEndpoints.js";
import {
    formatDate,
    formatProcessingTime,
    formatBytes,
} from "../utils/formatters.js";
import { showToast } from "../utils/toast.js";

// A detailed card for displaying a single model's complete analysis
const AnalysisResultCard = ({ analysis, videoId }) => {
    const isReal = analysis.prediction === "REAL";
    const confidence = analysis.confidence * 100;
    const modelInfo = MODEL_INFO[analysis.model];
    const detailsLink = `/results/${videoId}/${analysis.id}`;

    return (
        <Card
            className={`border-2 ${
                isReal ? "border-green-500/30" : "border-red-500/30"
            } overflow-hidden`}
        >
            <div className="p-6">
                <div className="flex justify-between items-start mb-4">
                    <div className="flex items-center gap-4">
                        <div className="p-3 bg-light-muted-background dark:bg-dark-muted-background rounded-lg">
                            <Brain className="w-6 h-6 text-primary-main" />
                        </div>
                        <div>
                            <h3 className="text-xl font-bold">
                                {modelInfo?.label || analysis.model}
                            </h3>
                            <p className="text-sm text-gray-500">
                                {modelInfo?.description}
                            </p>
                        </div>
                    </div>
                    <div
                        className={`p-2 rounded-lg ${
                            isReal
                                ? "bg-green-100 dark:bg-green-900/30"
                                : "bg-red-100 dark:bg-red-900/30"
                        }`}
                    >
                        {isReal ? (
                            <ShieldCheck className="h-7 w-7 text-green-600" />
                        ) : (
                            <ShieldAlert className="h-7 w-7 text-red-600" />
                        )}
                    </div>
                </div>

                <div className="grid grid-cols-2 gap-4 text-center my-6">
                    <div>
                        <p className="text-4xl font-bold">
                            {confidence.toFixed(1)}%
                        </p>
                        <p className="text-xs text-gray-500">Confidence</p>
                    </div>
                    <div>
                        <p
                            className={`text-4xl font-bold ${
                                isReal ? "text-green-600" : "text-red-600"
                            }`}
                        >
                            {analysis.prediction}
                        </p>
                        <p className="text-xs text-gray-500">Prediction</p>
                    </div>
                </div>

                <div className="text-sm space-y-2 text-gray-700 dark:text-gray-300">
                    <div className="flex justify-between">
                        <span>
                            <Clock className="inline w-4 h-4 mr-2 opacity-70" />
                            Processing Time:
                        </span>{" "}
                        <span className="font-mono">
                            {formatProcessingTime(analysis.processingTime)}
                        </span>
                    </div>
                    <div className="flex justify-between">
                        <span>
                            <Activity className="inline w-4 h-4 mr-2 opacity-70" />
                            Analysis Type:
                        </span>{" "}
                        <span className="font-mono">
                            {analysis.analysisType}
                        </span>
                    </div>
                </div>

                <Link to={detailsLink} className="mt-6 block">
                    <Button variant="outline" className="w-full">
                        View Full Forensic Report
                    </Button>
                </Link>
            </div>
            {analysis.status === "FAILED" && (
                <div className="bg-red-50 dark:bg-red-900/20 p-4 border-t border-red-200 dark:border-red-700">
                    <div className="flex items-center gap-2 text-red-600">
                        <AlertTriangle className="h-5 w-5" />
                        <h4 className="font-bold">Analysis Failed</h4>
                    </div>
                    <p className="text-xs text-red-700 dark:text-red-300 mt-1">
                        {analysis.errorMessage}
                    </p>
                </div>
            )}
        </Card>
    );
};

const Results = () => {
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

    if (isLoading) return <PageLoader text="Loading Analysis Report..." />;

    if (error)
        return (
            <div className="text-center p-8">
                <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
                <h2 className="text-2xl font-bold mb-2">Error Loading Video</h2>
                <p className="mb-6">{error.message}</p>
                <Link to="/dashboard">
                    <Button>
                        <ArrowLeft className="mr-2 h-4 w-4" />
                        Back to Dashboard
                    </Button>
                </Link>
            </div>
        );

    if (!video) return <div>Video not found.</div>;

    const completedAnalyses =
        video.analyses?.filter((a) => a.status === "COMPLETED") || [];
    const failedAnalyses =
        video.analyses?.filter((a) => a.status === "FAILED") || [];

    if (["QUEUED", "PROCESSING"].includes(video.status)) {
        return <AnalysisInProgress video={video} />;
    }

    return (
        <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Left Column: Video Details & Actions */}
                <div className="lg:col-span-1 space-y-6">
                    <Card>
                        <div className="p-4 border-b dark:border-gray-700">
                            <h3 className="font-semibold flex items-center gap-2">
                                <VideoIcon className="w-5 h-5 text-primary-main" />
                                Video Details
                            </h3>
                        </div>
                        <div className="p-4 space-y-4">
                            <VideoPlayer videoUrl={video.url} />
                            <div>
                                <h4 className="font-bold text-lg">
                                    {video.filename}
                                </h4>
                                {video.description && (
                                    <p className="text-sm text-gray-500 mt-1">
                                        {video.description}
                                    </p>
                                )}
                            </div>
                            <div className="text-xs space-y-2 text-gray-600 dark:text-gray-400">
                                <div className="flex justify-between">
                                    <span>
                                        <FileText className="inline w-4 h-4 mr-1" />
                                        MIME Type:
                                    </span>{" "}
                                    <span className="font-mono">
                                        {video.mimetype}
                                    </span>
                                </div>
                                <div className="flex justify-between">
                                    <span>
                                        <HardDrive className="inline w-4 h-4 mr-1" />
                                        File Size:
                                    </span>{" "}
                                    <span className="font-mono">
                                        {formatBytes(video.size)}
                                    </span>
                                </div>
                            </div>
                        </div>
                    </Card>

                    <Card>
                        <div className="p-4 border-b dark:border-gray-700">
                            <h3 className="font-semibold flex items-center gap-2">
                                <Activity className="w-5 h-5 text-primary-main" />
                                Actions
                            </h3>
                        </div>
                        <div className="p-4 space-y-3">
                            <Button
                                onClick={handleDownloadPDF}
                                variant="outline"
                                className="w-full"
                                disabled={isDownloadingPDF}
                            >
                                {isDownloadingPDF ? (
                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                ) : (
                                    <FileDown className="mr-2 h-4 w-4" />
                                )}
                                {isDownloadingPDF
                                    ? "Generating..."
                                    : "Download PDF Report"}
                            </Button>
                            <Button
                                onClick={() =>
                                    setModal({
                                        type: "new_analysis",
                                        data: video,
                                    })
                                }
                                className="w-full"
                                variant="outline"
                            >
                                <Brain className="mr-2 h-4 w-4" />
                                Run New Analysis
                            </Button>
                            <Button
                                onClick={() =>
                                    setModal({ type: "edit", data: video })
                                }
                                className="w-full"
                                variant="outline"
                            >
                                <Edit className="mr-2 h-4 w-4" />
                                Edit Details
                            </Button>
                            <Button
                                onClick={() =>
                                    setModal({ type: "delete", data: video })
                                }
                                className="w-full"
                                variant="destructive"
                            >
                                <Trash2 className="mr-2 h-4 w-4" />
                                Delete Video
                            </Button>
                        </div>
                    </Card>
                </div>

                {/* Right Column: Analysis Results */}
                <div className="lg:col-span-2 space-y-6">
                    {completedAnalyses.length === 0 &&
                    failedAnalyses.length === 0 ? (
                        <Card className="flex flex-col items-center justify-center text-center p-12">
                            <Brain className="w-16 h-16 text-gray-300 dark:text-gray-600 mb-4" />
                            <h3 className="text-xl font-bold">
                                No Analysis Results
                            </h3>
                            <p className="text-gray-500 mb-4">
                                This video has not been analyzed yet.
                            </p>
                            <Button
                                onClick={() =>
                                    setModal({
                                        type: "new_analysis",
                                        data: video,
                                    })
                                }
                            >
                                Start First Analysis
                            </Button>
                        </Card>
                    ) : (
                        [...completedAnalyses, ...failedAnalyses]
                            .sort(
                                (a, b) =>
                                    new Date(b.createdAt) -
                                    new Date(a.createdAt)
                            )
                            .map((analysis) => (
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
                onAnalysisStart={() => {
                    refetch();
                }}
            />
        </div>
    );
};

export default Results;
