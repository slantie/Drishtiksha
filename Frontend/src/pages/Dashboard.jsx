import React, { useState, useEffect, useCallback } from "react";
import {
    Upload,
    Play,
    Clock,
    CheckCircle,
    XCircle,
    AlertCircle,
    Eye,
    BarChart3,
    Loader2,
} from "lucide-react";
import { showToast } from "../utils/toast";

const backendUrl = import.meta.env.VITE_BACKEND_URL || "http://localhost:4000";

export const Dashboard = () => {
    const [videos, setVideos] = useState([]);
    const [loading, setLoading] = useState(true);
    const [selectedVideo, setSelectedVideo] = useState(null);
    const [analyzing, setAnalyzing] = useState({});

    const getAuthToken = () => {
        return (
            localStorage.getItem("authToken") ||
            sessionStorage.getItem("authToken")
        );
    };

    const fetchVideos = useCallback(async () => {
        setLoading(true);
        try {
            const token = getAuthToken();
            const response = await fetch(`${backendUrl}/api/video`, {
                headers: {
                    Authorization: `Bearer ${token}`,
                },
            });

            const data = await response.json();
            if (data.success) {
                setVideos(data.data);
            } else {
                showToast.error("Failed to load videos");
            }
        } catch (error) {
            console.error("Error fetching videos:", error);
            showToast.error("Failed to load videos");
        } finally {
            setLoading(false);
        }
    }, []); // Empty dependency array: function is created only once.

    useEffect(() => {
        fetchVideos();
    }, [fetchVideos]); // This now works correctly.

    const analyzeVideo = async (videoId) => {
        const token = getAuthToken();
        if (!token) {
            showToast.error("Authentication required");
            return;
        }

        setAnalyzing((prev) => ({ ...prev, [videoId]: true }));
        const loadingToast = showToast.loading(
            "Analyzing video for deepfakes..."
        );

        try {
            const response = await fetch(
                `${backendUrl}/api/video/${videoId}/analyze`,
                {
                    method: "POST",
                    headers: {
                        Authorization: `Bearer ${token}`,
                    },
                }
            );

            const data = await response.json();
            showToast.dismiss(loadingToast);

            if (data.success) {
                showToast.success("Analysis completed successfully!");
                fetchVideos(); // Refresh the list
            } else {
                showToast.error(data.message || "Analysis failed");
            }
        } catch (error) {
            showToast.dismiss(loadingToast);
            console.error("Analysis error:", error);
            showToast.error("Analysis failed. Please try again.");
        } finally {
            setAnalyzing((prev) => ({ ...prev, [videoId]: false }));
        }
    };

    const getStatusIcon = (status) => {
        switch (status) {
            case "UPLOADED":
                return <Clock className="w-5 h-5 text-blue-500" />;
            case "PROCESSING":
                return (
                    <Loader2 className="w-5 h-5 text-yellow-500 animate-spin" />
                );
            case "ANALYZED":
                return <CheckCircle className="w-5 h-5 text-green-500" />;
            case "FAILED":
                return <XCircle className="w-5 h-5 text-red-500" />;
            default:
                return <AlertCircle className="w-5 h-5 text-gray-500" />;
        }
    };

    const getStatusColor = (status) => {
        switch (status) {
            case "UPLOADED":
                return "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200";
            case "PROCESSING":
                return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200";
            case "ANALYZED":
                return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200";
            case "FAILED":
                return "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200";
            default:
                return "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200";
        }
    };

    const getPredictionBadge = (analysis) => {
        if (!analysis) return null;

        const isReal = analysis.prediction === "REAL";
        const confidence = (analysis.confidence * 100).toFixed(1);

        return (
            <div
                className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                    isReal
                        ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                        : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
                }`}
            >
                {isReal ? "✓ Real" : "⚠ Deepfake"} ({confidence}%)
            </div>
        );
    };

    // **FIXED**: Mapping object for stat icon colors
    const iconColorMap = {
        blue: "text-blue-500",
        green: "text-green-500",
        yellow: "text-yellow-500",
        purple: "text-purple-500",
    };

    if (loading) {
        return (
            <div className="flex justify-center items-center h-64">
                <Loader2 className="w-8 h-8 animate-spin text-light-highlight dark:text-dark-highlight" />
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-light-text dark:text-dark-text">
                        Video Analysis Dashboard
                    </h1>
                    <p className="text-light-muted-text dark:text-dark-muted-text">
                        Monitor and analyze your uploaded videos for deepfake
                        detection
                    </p>
                </div>
                <a
                    href="/upload"
                    className="flex items-center space-x-2 bg-light-highlight dark:bg-dark-highlight text-white px-6 py-3 rounded-xl hover:shadow-lg transition-all duration-300"
                >
                    <Upload className="w-5 h-5" />
                    <span>Upload Video</span>
                </a>
            </div>

            {/* Statistics */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                {[
                    {
                        label: "Total Videos",
                        value: videos.length,
                        icon: Play,
                        color: "blue",
                    },
                    {
                        label: "Analyzed",
                        value: videos.filter((v) => v.status === "ANALYZED")
                            .length,
                        icon: CheckCircle,
                        color: "green",
                    },
                    {
                        label: "Processing",
                        value: videos.filter((v) => v.status === "PROCESSING")
                            .length,
                        icon: Clock,
                        color: "yellow",
                    },
                    {
                        label: "Real Videos",
                        value: videos.filter(
                            (v) => v.analysis?.prediction === "REAL"
                        ).length,
                        icon: Eye,
                        color: "purple",
                    },
                ].map((stat, index) => (
                    <div
                        key={index}
                        className="bg-light-background dark:bg-dark-background p-6 rounded-xl border border-light-muted-text/10 dark:border-dark-muted-text/10"
                    >
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-light-muted-text dark:text-dark-muted-text text-sm">
                                    {stat.label}
                                </p>
                                <p className="text-2xl font-bold text-light-text dark:text-dark-text">
                                    {stat.value}
                                </p>
                            </div>
                            {/* **FIXED**: Using the color map to apply the correct class */}
                            <stat.icon
                                className={`w-8 h-8 ${
                                    iconColorMap[stat.color]
                                }`}
                            />
                        </div>
                    </div>
                ))}
            </div>

            {/* Videos List */}
            <div className="bg-light-background dark:bg-dark-background rounded-xl border border-light-muted-text/10 dark:border-dark-muted-text/10 overflow-hidden">
                <div className="p-6 border-b border-light-muted-text/10 dark:border-dark-muted-text/10">
                    <h2 className="text-xl font-semibold text-light-text dark:text-dark-text">
                        Your Videos
                    </h2>
                </div>

                {videos.length === 0 ? (
                    <div className="p-12 text-center">
                        <Upload className="w-16 h-16 text-light-muted-text dark:text-dark-muted-text mx-auto mb-4" />
                        <h3 className="text-lg font-medium text-light-text dark:text-dark-text mb-2">
                            No videos uploaded
                        </h3>
                        <p className="text-light-muted-text dark:text-dark-muted-text mb-6">
                            Start by uploading your first video for deepfake
                            analysis
                        </p>
                        <a
                            href="/upload"
                            className="inline-flex items-center space-x-2 bg-light-highlight dark:bg-dark-highlight text-white px-6 py-3 rounded-xl hover:shadow-lg transition-all duration-300"
                        >
                            <Upload className="w-5 h-5" />
                            <span>Upload Video</span>
                        </a>
                    </div>
                ) : (
                    <div className="divide-y divide-light-muted-text/10 dark:divide-dark-muted-text/10">
                        {videos.map((video) => (
                            <div
                                key={video.id}
                                className="p-6 hover:bg-light-muted-background/50 dark:hover:bg-dark-muted-background/50 transition-colors"
                            >
                                <div className="flex items-center justify-between">
                                    <div className="flex-1">
                                        <div className="flex items-center space-x-3 mb-2">
                                            <h3 className="text-lg font-medium text-light-text dark:text-dark-text">
                                                {video.filename}
                                            </h3>
                                            {getStatusIcon(video.status)}
                                            <span
                                                className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(
                                                    video.status
                                                )}`}
                                            >
                                                {video.status}
                                            </span>
                                        </div>

                                        {video.description && (
                                            <p className="text-light-muted-text dark:text-dark-muted-text mb-2">
                                                {video.description}
                                            </p>
                                        )}

                                        <div className="flex items-center space-x-4 text-sm text-light-muted-text dark:text-dark-muted-text">
                                            <span>
                                                Size:{" "}
                                                {(
                                                    video.size /
                                                    (1024 * 1024)
                                                ).toFixed(2)}{" "}
                                                MB
                                            </span>
                                            <span>
                                                Uploaded:{" "}
                                                {new Date(
                                                    video.createdAt
                                                ).toLocaleDateString()}
                                            </span>
                                            {video.analysis && (
                                                <span>
                                                    Processed in:{" "}
                                                    {video.analysis.processingTime?.toFixed(
                                                        2
                                                    )}
                                                    s
                                                </span>
                                            )}
                                        </div>

                                        {video.analysis && (
                                            <div className="mt-3">
                                                {getPredictionBadge(
                                                    video.analysis
                                                )}
                                            </div>
                                        )}
                                    </div>

                                    <div className="flex items-center space-x-3">
                                        {video.status === "UPLOADED" && (
                                            <button
                                                onClick={() =>
                                                    analyzeVideo(video.id)
                                                }
                                                disabled={analyzing[video.id]}
                                                className="flex items-center space-x-2 bg-light-highlight dark:bg-dark-highlight text-white px-4 py-2 rounded-lg hover:shadow-lg transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
                                            >
                                                {analyzing[video.id] ? (
                                                    <Loader2 className="w-4 h-4 animate-spin" />
                                                ) : (
                                                    <BarChart3 className="w-4 h-4" />
                                                )}
                                                <span>
                                                    {analyzing[video.id]
                                                        ? "Analyzing..."
                                                        : "Analyze"}
                                                </span>
                                            </button>
                                        )}

                                        {video.analysis && (
                                            <button
                                                onClick={() =>
                                                    setSelectedVideo(video)
                                                }
                                                className="flex items-center space-x-2 bg-light-muted-background dark:bg-dark-muted-background text-light-text dark:text-dark-text px-4 py-2 rounded-lg hover:shadow-lg transition-all duration-300"
                                            >
                                                <Eye className="w-4 h-4" />
                                                <span>View Details</span>
                                            </button>
                                        )}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Video Details Modal */}
            {selectedVideo && (
                <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
                    <div className="bg-light-background dark:bg-dark-background rounded-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
                        <div className="p-6 border-b border-light-muted-text/10 dark:border-dark-muted-text/10">
                            <div className="flex justify-between items-center">
                                <h2 className="text-xl font-semibold text-light-text dark:text-dark-text">
                                    Analysis Details
                                </h2>
                                <button
                                    onClick={() => setSelectedVideo(null)}
                                    className="text-light-muted-text dark:text-dark-muted-text hover:text-light-text dark:hover:text-dark-text"
                                    aria-label="Close modal" // Accessibility improvement
                                >
                                    ✕
                                </button>
                            </div>
                        </div>

                        <div className="p-6 space-y-6">
                            <div>
                                <h3 className="font-medium text-light-text dark:text-dark-text mb-2">
                                    Video Information
                                </h3>
                                <div className="space-y-2 text-sm">
                                    <div className="flex justify-between">
                                        <span className="text-light-muted-text dark:text-dark-muted-text">
                                            Filename:
                                        </span>
                                        <span className="text-light-text dark:text-dark-text">
                                            {selectedVideo.filename}
                                        </span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-light-muted-text dark:text-dark-muted-text">
                                            Size:
                                        </span>
                                        <span className="text-light-text dark:text-dark-text">
                                            {(
                                                selectedVideo.size /
                                                (1024 * 1024)
                                            ).toFixed(2)}{" "}
                                            MB
                                        </span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-light-muted-text dark:text-dark-muted-text">
                                            Uploaded:
                                        </span>
                                        <span className="text-light-text dark:text-dark-text">
                                            {new Date(
                                                selectedVideo.createdAt
                                            ).toLocaleString()}
                                        </span>
                                    </div>
                                </div>
                            </div>

                            {selectedVideo.analysis && (
                                <div>
                                    <h3 className="font-medium text-light-text dark:text-dark-text mb-2">
                                        Analysis Results
                                    </h3>
                                    <div className="space-y-3">
                                        <div className="flex justify-between items-center">
                                            <span className="text-light-muted-text dark:text-dark-muted-text">
                                                Prediction:
                                            </span>
                                            {getPredictionBadge(
                                                selectedVideo.analysis
                                            )}
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-light-muted-text dark:text-dark-muted-text">
                                                Confidence:
                                            </span>
                                            <span className="text-light-text dark:text-dark-text">
                                                {(
                                                    selectedVideo.analysis
                                                        .confidence * 100
                                                ).toFixed(2)}
                                                %
                                            </span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-light-muted-text dark:text-dark-muted-text">
                                                Processing Time:
                                            </span>
                                            <span className="text-light-text dark:text-dark-text">
                                                {selectedVideo.analysis.processingTime?.toFixed(
                                                    2
                                                )}
                                                s
                                            </span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-light-muted-text dark:text-dark-muted-text">
                                                Model Version:
                                            </span>
                                            <span className="text-light-text dark:text-dark-text">
                                                {
                                                    selectedVideo.analysis
                                                        .modelVersion
                                                }
                                            </span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-light-muted-text dark:text-dark-muted-text">
                                                Analysis Date:
                                            </span>
                                            <span className="text-light-text dark:text-dark-text">
                                                {new Date(
                                                    selectedVideo.analysis.createdAt
                                                ).toLocaleString()}
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};
