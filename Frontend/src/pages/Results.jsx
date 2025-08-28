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
    useMediaItemQuery,
    useUpdateMediaMutation,
    useDeleteMediaMutation,
} from "../hooks/useMediaQuery.jsx";
import { AuthContext } from "../contexts/AuthContext.jsx";
import { useAuth } from "../contexts/AuthContext.jsx";
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
import { MediaPlayer } from "../components/media/MediaPlayer.jsx"; 
import { AnalysisInProgress } from "../components/media/AnalysisInProgress.jsx";
import { EditMediaModal } from "../components/media/EditMediaModal.jsx";
import { DeleteMediaModal } from "../components/media/DeleteMediaModal.jsx";
import ModelSelectionModal from "../components/analysis/ModelSelectionModal.jsx";
import { DownloadService } from "../services/DownloadReport.js";
import { MODEL_INFO } from "../constants/apiEndpoints.js";
import { formatProcessingTime, formatBytes } from "../utils/formatters.js";
import { showToast } from "../utils/toast.js";
import { PageHeader } from "../components/layout/PageHeader.jsx";
import { useServerStatusQuery } from "../hooks/useMonitoringQuery";

import {
    Alert,
    AlertTitle,
    AlertDescription,
} from "../components/ui/Alert.jsx";
import { EmptyState } from "../components/ui/EmptyState.jsx";
import { useMemo } from "react";

// REFACTOR: The result card is now cleaner and uses Card sub-components for better structure.
const AnalysisResultCard = ({ analysis, mediaId, serverModels }) => {
    // 1. Find the full model information from the server status data
    //    by matching the model name from the analysis record.
    const modelInfo = useMemo(() => {
        return serverModels?.find(
            (m) => m.name.toLowerCase() === analysis.model.toLowerCase()
        );
    }, [serverModels, analysis.model]);

    // 2. Create fallback data for the UI in case the model is no longer active
    const displayName = modelInfo?.name || analysis.model;
    const displayDescription =
        modelInfo?.description || "Detailed analysis results.";

    console.log("Models:", serverModels);
    console.log("Model Info:", modelInfo);
    console.log("Analysis Data:", analysis.model);

    // 3. Handle the display for a FAILED analysis
    if (analysis.status === "FAILED") {
        return (
            <Card>
                <CardHeader>
                    <CardTitle>{displayName}</CardTitle>
                    <CardDescription>{displayDescription}</CardDescription>
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

    // 4. Render the successful analysis card
    const isReal = analysis.prediction === "REAL";
    const confidence = analysis.confidence * 100;

    return (
        <Card
            className={`transition-all hover:shadow-lg ${
                isReal ? "border-green-500/30" : "border-red-500/30"
            }`}
        >
            <CardHeader className="flex flex-row items-start justify-between">
                <div>
                    <CardTitle>{displayName}</CardTitle>
                    <CardDescription>{displayDescription}</CardDescription>
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
                    <Link
                        to={`/results/${mediaId}/${analysis.id}`}
                        className="flex items-center justify-center gap-2"
                    >
                        <ChartNetwork className="h-5 w-5 text-purple-500" />
                        View Detailed Report
                    </Link>
                </Button>
            </CardFooter>
        </Card>
    );
};

const Results = () => {
    const { videoId: mediaId } = useParams(); // Use alias for clarity
    const navigate = useNavigate();
    const { user } = useAuth();
    
    const { data: serverStatus } = useServerStatusQuery();
    const serverModels = serverStatus?.modelsInfo || [];

    const { data: media, isLoading, error, refetch } = useMediaItemQuery(mediaId, {
        refetchInterval: (query) => 
            ["QUEUED", "PROCESSING"].includes(query.state.data?.status) ? 5000 : false,
    });

    const updateMutation = useUpdateMediaMutation();
    const deleteMutation = useDeleteMediaMutation();

    const [modal, setModal] = useState({ type: null, data: null });
    const [isDownloadingPDF, setIsDownloadingPDF] = useState(false);
    const [isDownloadingMedia, setIsDownloadingMedia] = useState(false);

    const handleDownloadPDF = async () => {
        if (!media) return;
        setIsDownloadingPDF(true);
        try {
            await DownloadService.generateAndDownloadPDF(media, user);
        } catch (err) {
            showToast.error(err.message || "Failed to generate PDF report.");
        } finally {
            setIsDownloadingPDF(false);
        }
    };

    if (isLoading) return <PageLoader text="Loading Media Data" />;

    if (error)
        return (
            <div className="text-center py-16">
                <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
                <h2 className="text-2xl font-bold">Error Loading Media</h2>
                <p className="mb-6 text-light-muted-text dark:text-dark-muted-text">{error.message}</p>
                <Button asChild><Link to="/dashboard"><ArrowLeft className="mr-2 h-4 w-4" /> Go Back</Link></Button>
            </div>
        );
    
    // FIX: Add a robust check to ensure the media object exists before rendering.
    if (!media) return <PageLoader text="Media not found..." />;

    const isProcessing = ["QUEUED", "PROCESSING"].includes(media.status);
    
    const completedAnalyses = media.analyses?.filter((a) => a.status === "COMPLETED") || [];
    const failedAnalyses = media.analyses?.filter((a) => a.status === "FAILED") || [];
    const allAnalyses = [...completedAnalyses, ...failedAnalyses].sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));

    return (
        <div className="space-y-4">
            <PageHeader
                title="Analysis Results"
                description={`Detailed report for ${media.filename}`}
                actions={<Button asChild variant="outline"><Link to="/dashboard"><ArrowLeft className="mr-2 h-4 w-4" /> Go Back</Link></Button>}
            />
            
            {isProcessing ? (
                <AnalysisInProgress media={media} />
            ) : (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 items-start">
                    <div className="lg:col-span-1 space-y-4 sticky top-24">
                        <Card>
                            <CardHeader><CardTitle className="flex items-center gap-2"><VideoIcon className="w-5 h-5 text-primary-main" /> Media Details & Actions</CardTitle></CardHeader>
                            <CardContent className="space-y-4">
                                {/* FIX: Pass the entire 'media' object to the MediaPlayer component. */}
                                <MediaPlayer media={media} />
                                <div>
                                    <h4 className="font-semibold text-lg">{media.filename}</h4>
                                    <p className="text-sm text-light-muted-text dark:text-dark-muted-text mt-1">{media.description || "No description provided."}</p>
                                </div>
                                <Button
                                    onClick={() => {
                                        setIsDownloadingMedia(true);
                                        // FIX: Corrected function call from downloadMedia to downloadVideo
                                        DownloadService.downloadVideo(media.url, media.filename)
                                            .then(() => showToast.success("Media download started."))
                                            .catch((error) => showToast.error(`Failed to download media: ${error.message}`))
                                            .finally(() => setIsDownloadingMedia(false));
                                    }}
                                    variant="outline"
                                    isLoading={isDownloadingMedia}
                                    className="w-full"
                                >
                                    <Download className="mr-2 h-4 w-4" />
                                    {isDownloadingMedia ? "Preparing..." : "Download Media"}
                                </Button>
                            </CardContent>
                            <CardFooter className="grid grid-cols-2 gap-2">
                                <Button onClick={() => setModal({ type: "new_analysis", data: media })} variant="outline"><Brain className="h-4 w-4 mr-2" /> Re-Analyze</Button>
                                <Button onClick={() => setModal({ type: "edit", data: media })} variant="outline"><Edit className="h-4 w-4 mr-2" /> Edit</Button>
                                <Button onClick={handleDownloadPDF} variant="outline" isLoading={isDownloadingPDF}><FileSpreadsheetIcon className="h-4 w-4 mr-2" /> PDF Report</Button>
                                <Button onClick={() => setModal({ type: "delete", data: media })} variant="destructive"><Trash2 className="h-4 w-4 mr-2" /> Delete</Button>
                            </CardFooter>
                        </Card>
                    </div>

                    <div className="lg:col-span-2 space-y-4">
                        {allAnalyses.length === 0 ? (
                            <EmptyState
                                icon={Bot}
                                title="No Analysis Results"
                                message="This media is awaiting its first analysis. Run one to get started."
                                action={<Button onClick={() => setModal({ type: "new_analysis", data: media })}><Brain className="mr-2 h-4 w-4" /> Start First Analysis</Button>}
                            />
                        ) : (
                            allAnalyses.map((analysis) => (
                                <AnalysisResultCard key={analysis.id} analysis={analysis} mediaId={media.id} serverModels={serverModels} />
                            ))
                        )}
                    </div>
                </div>
            )}

            {/* Modals */}
            <EditMediaModal
                isOpen={modal.type === "edit"}
                onClose={() => setModal({ type: null })}
                // FIX: Pass the 'media' prop instead of 'video'.
                media={modal.data}
                onUpdate={(id, data) => updateMutation.mutateAsync({ mediaId: id, updateData: data })}
            />
            <DeleteMediaModal
                isOpen={modal.type === "delete"}
                onClose={() => setModal({ type: null })}
                // FIX: Pass the 'media' prop instead of 'video'.
                media={modal.data}
                onDelete={(id) => deleteMutation.mutateAsync(id).then(() => navigate("/dashboard"))}
            />
            <ModelSelectionModal
                isOpen={modal.type === "new_analysis"}
                onClose={() => setModal({ type: null })}
                // Note: This prop is 'videoId' in the modal, should be renamed to 'mediaId' there for consistency later.
                videoId={modal.data?.id}
                onAnalysisStart={refetch}
            />
        </div>
    );
};


export default Results;
