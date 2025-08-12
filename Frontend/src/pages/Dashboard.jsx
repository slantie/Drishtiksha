/**
 * @file src/pages/dashboard.jsx
 * @description Main dashboard page for video analysis, featuring a data table, stat cards, and video analysis controls, now with an integrated upload modal.
 */

import React, { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { useNavigate } from "react-router-dom";
import {
    Upload, Play, CheckCircle, RefreshCw, Clock, MoreHorizontal, AlertTriangle, Loader2,
    UploadCloud, X, FileVideo, Edit, Trash2, Save
} from "lucide-react";
import { StatCard } from "../components/ui/StatCard";
import { DataTable } from "../components/ui/DataTable";
import { PageLoader } from "../components/ui/LoadingSpinner";
import { showToast } from "../utils/toast";
import { API_ENDPOINTS } from "../constants/apiEndpoints";
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger
} from "../components/ui/DropdownMenu";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";

const ANALYSIS_MODELS = ["SIGLIPV1", "RPPG", "COLORCUES"];

// --- Helper functions for formatting data ---
/**
 * Formats a number of bytes into a human-readable string.
 * @param {number} bytes - The size in bytes.
 * @param {number} [decimals=2] - The number of decimal places to use.
 * @returns {string} - The formatted size string.
 */
const formatBytes = (bytes, decimals = 2) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
};

/**
 * Formats an ISO date string into a user-friendly date format.
 * @param {string} isoDate - The ISO 8601 date string.
 * @returns {string} - The formatted date string.
 */
const formatDate = (isoDate) => {
    const date = new Date(isoDate);
    const options = { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' };
    return date.toLocaleDateString('en-US', options);
};

// --- Reusable Sub-components for DataTable's `accessor` prop ---
/**
 * Renders a status badge with a color-coded background and an icon.
 * @param {object} props - The component props.
 * @param {string} props.status - The status string (e.g., "ANALYZED", "PROCESSING").
 */
const StatusBadge = ({ status }) => {
    const styles = {
        ANALYZED: "bg-green-500/10 text-green-500",
        PROCESSING: "bg-yellow-500/10 text-yellow-500",
        UPLOADED: "bg-blue-500/10 text-blue-500",
        FAILED: "bg-red-500/10 text-red-500",
    };
    const Icon = { ANALYZED: CheckCircle, PROCESSING: Loader2, UPLOADED: Clock, FAILED: AlertTriangle }[status];
    return (
        <div className={`inline-flex items-center gap-2 text-xs font-semibold px-3 py-1.5 rounded-full ${styles[status]}`}>
            <Icon className={`w-4 h-4 ${status === 'PROCESSING' ? 'animate-spin' : ''}`} />
            <span>{status}</span>
        </div>
    );
};

/**
 * Renders a progress bar indicating the number of completed analyses.
 * @param {object} props - The component props.
 * @param {Array<object>} props.analyses - The array of analysis objects.
 */
const ProgressIndicator = ({ analyses }) => {
    const percentage = (analyses.length / ANALYSIS_MODELS.length) * 100;
    return (
        <div className="w-full max-w-[150px]">
            <span className="text-sm font-semibold text-light-text dark:text-dark-text">{analyses.length}/{ANALYSIS_MODELS.length} Models</span>
            <div className="w-full bg-light-secondary dark:bg-dark-secondary rounded-full h-2 mt-1">
                <div className="bg-primary-main h-2 rounded-full" style={{ width: `${percentage}%` }}/>
            </div>
        </div>
    );
};

// --- New Upload Modal Component ---
const UploadModal = ({ isOpen, onClose, onUploadSuccess }) => {
    const [videoFile, setVideoFile] = useState(null);
    const [description, setDescription] = useState("");
    const [isUploading, setIsUploading] = useState(false);
    const [dragActive, setDragActive] = useState(false);
    const fileInputRef = useRef(null);

    const getAuthToken = () => {
        return (
            localStorage.getItem("authToken") ||
            sessionStorage.getItem("authToken")
        );
    };

    const clearVideo = useCallback(() => {
        setVideoFile(null);
        setDescription("");
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    }, []);

    const handleFileSelect = (file) => {
        if (file && file.type.startsWith("video/")) {
            setVideoFile(file);
        } else {
            showToast.error("Please select a valid video file.");
            clearVideo();
        }
    };

    const handleFileInputChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileSelect(file);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setDragActive(false);
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        setDragActive(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        setDragActive(false);
    };

    const formatFileSize = (bytes) => {
        if (bytes === 0) return "0 Bytes";
        const k = 1024;
        const sizes = ["Bytes", "KB", "MB", "GB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        const token = getAuthToken();
        if (!token) {
            showToast.error("You must be logged in to upload videos.");
            return;
        }

        if (!videoFile) {
            showToast.warning("Please select a video file to upload.");
            return;
        }

        setIsUploading(true);
        const loadingToast = showToast.loading("Uploading video...");

        const formData = new FormData();
        formData.append("video", videoFile);
        formData.append("description", description);

        try {
            const res = await fetch(API_ENDPOINTS.UPLOAD_VIDEO, {
                method: "POST",
                headers: {
                    Authorization: `Bearer ${token}`,
                },
                body: formData,
            });

            const data = await res.json();
            showToast.dismiss(loadingToast);

            if (res.ok && data.success) {
                showToast.success(
                    "Video uploaded successfully! It is now being analyzed."
                );
                clearVideo();
                onUploadSuccess(); // Close modal and refresh dashboard data
            } else {
                showToast.error(
                    data.message || "Upload failed. Please try again."
                );
            }
        } catch (err) {
            showToast.dismiss(loadingToast);
            console.error("Upload error:", err);
            showToast.error(
                "An error occurred while uploading. Please check your connection and try again."
            );
        } finally {
            setIsUploading(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-light-background/80 dark:bg-dark-background/80 backdrop-blur-sm transition-opacity duration-300">
            <div className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto bg-light-background dark:bg-dark-background rounded-3xl p-8 shadow-2xl border border-light-muted-text/10 dark:border-dark-muted-text/10">
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 p-2 rounded-full text-light-muted-text dark:text-dark-muted-text hover:bg-light-muted-background/50 dark:hover:bg-dark-muted-background/50 transition-colors"
                >
                    <X className="w-5 h-5" />
                </button>
                <div className="flex items-center space-x-4 mb-8">
                    <div className="p-3 bg-light-highlight/10 dark:bg-dark-highlight/10 rounded-2xl">
                        <Upload className="w-8 h-8 text-light-highlight dark:text-dark-highlight" />
                    </div>
                    <div>
                        <h2 className="text-2xl font-bold text-light-text dark:text-dark-text">
                            Upload Video
                        </h2>
                        <p className="text-light-muted-text dark:text-dark-muted-text">
                            Select a video file to begin analysis
                        </p>
                    </div>
                </div>

                <form onSubmit={handleSubmit} className="space-y-6">
                    <div>
                        <label className="block text-base font-semibold text-light-text dark:text-dark-text mb-4">
                            Select Video File
                        </label>
                        <div
                            className={`relative border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-300 cursor-pointer ${
                                dragActive
                                    ? "border-light-highlight dark:border-dark-highlight bg-light-highlight/5 dark:bg-dark-highlight/5 scale-[1.01]"
                                    : "border-light-muted-text/30 dark:border-dark-muted-text/30 hover:border-light-highlight/50 dark:hover:border-dark-highlight/50"
                            }`}
                            onDrop={handleDrop}
                            onDragOver={handleDragOver}
                            onDragLeave={handleDragLeave}
                            onClick={() => fileInputRef.current?.click()}
                        >
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept="video/*"
                                onChange={handleFileInputChange}
                                className="hidden"
                                disabled={isUploading}
                            />
                            {videoFile ? (
                                <div className="space-y-4">
                                    <CheckCircle className="w-16 h-16 text-green-500 mx-auto" />
                                    <div>
                                        <p className="text-xl font-semibold text-light-text dark:text-dark-text">
                                            {videoFile.name}
                                        </p>
                                        <p className="text-base text-light-muted-text dark:text-dark-muted-text mt-2">
                                            {formatFileSize(videoFile.size)}
                                        </p>
                                    </div>
                                    <button
                                        type="button"
                                        onClick={(e) => { e.stopPropagation(); clearVideo(); }}
                                        className="text-base text-light-highlight dark:text-dark-highlight hover:underline font-medium"
                                    >
                                        Choose different file
                                    </button>
                                </div>
                            ) : (
                                <div className="space-y-4">
                                    <FileVideo className="w-16 h-16 text-light-muted-text dark:text-dark-muted-text mx-auto" />
                                    <div>
                                        <p className="text-xl font-semibold text-light-text dark:text-dark-text">
                                            Drag & drop your video here
                                        </p>
                                        <p className="text-base text-light-muted-text dark:text-dark-muted-text mt-2">
                                            or click to browse files
                                        </p>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                    <div>
                        <label className="block text-base font-semibold text-light-text dark:text-dark-text mb-4">
                            Description (Optional)
                        </label>
                        <textarea
                            placeholder="Add a detailed description for your video analysis..."
                            value={description}
                            onChange={(e) => setDescription(e.target.value)}
                            className="w-full px-6 py-4 bg-light-muted-background dark:bg-dark-muted-background border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-2xl text-light-text dark:text-dark-text focus:outline-none focus:ring-3 focus:ring-light-highlight/20 dark:focus:ring-dark-highlight/20 transition-all resize-none"
                            rows={3}
                            disabled={isUploading}
                        />
                    </div>
                    <Button
                        type="submit"
                        disabled={isUploading || !videoFile}
                        className="w-full bg-gradient-to-r from-light-highlight via-light-highlight to-light-highlight/80 dark:from-dark-highlight dark:via-dark-highlight dark:to-dark-highlight/80 text-white font-bold py-5 px-8 rounded-2xl shadow-xl hover:shadow-2xl transition-all duration-300 hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 flex items-center justify-center space-x-4 text-lg"
                    >
                        {isUploading ? (
                            <>
                                <div className="w-6 h-6 border-3 border-white/30 border-t-white rounded-full animate-spin"></div>
                                <span>Uploading & Processing...</span>
                            </>
                        ) : (
                            <>
                                <Upload className="w-6 h-6" />
                                <span>Upload & Start Analysis</span>
                            </>
                        )}
                    </Button>
                </form>
            </div>
        </div>
    );
};

// --- Edit Video Modal Component ---
const EditVideoModal = ({ isOpen, onClose, video, onUpdateSuccess }) => {
    const [description, setDescription] = useState(video?.description || "");
    const [filename, setFilename] = useState(video?.filename || "");
    const [isSaving, setIsSaving] = useState(false);

    useEffect(() => {
        if (video) {
            setDescription(video.description || "");
            setFilename(video.filename || "");
        }
    }, [video]);

    const getAuthToken = () => {
        return localStorage.getItem("authToken") || sessionStorage.getItem("authToken");
    };

    const handleSave = async () => {
        if (!video) return;

        setIsSaving(true);
        const loadingToast = showToast.loading("Updating video details...");

        try {
            const token = getAuthToken();
            const response = await fetch(API_ENDPOINTS.UPDATE_VIDEO(video.id), {
                method: "PATCH",
                headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${token}`,
                },
                body: JSON.stringify({ description, filename }),
            });

            const result = await response.json();
            showToast.dismiss(loadingToast);

            if (response.ok && result.success) {
                showToast.success("Video details updated successfully.");
                onUpdateSuccess();
            } else {
                showToast.error(result.message || "Failed to update video details.");
            }
        } catch (error) {
            showToast.dismiss(loadingToast);
            showToast.error("An error occurred while updating the video.");
        } finally {
            setIsSaving(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-light-background/80 dark:bg-dark-background/80 backdrop-blur-sm transition-opacity duration-300">
            <div className="relative w-full max-w-lg max-h-[90vh] overflow-y-auto bg-light-background dark:bg-dark-background rounded-3xl p-8 shadow-2xl border border-light-muted-text/10 dark:border-dark-muted-text/10">
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 p-2 rounded-full text-light-muted-text dark:text-dark-muted-text hover:bg-light-muted-background/50 dark:hover:bg-dark-muted-background/50 transition-colors"
                >
                    <X className="w-5 h-5" />
                </button>
                <div className="flex items-center space-x-4 mb-8">
                    <div className="p-3 bg-light-highlight/10 dark:bg-dark-highlight/10 rounded-2xl">
                        <Edit className="w-8 h-8 text-light-highlight dark:text-dark-highlight" />
                    </div>
                    <div>
                        <h2 className="text-2xl font-bold text-light-text dark:text-dark-text">
                            Edit Video
                        </h2>
                        <p className="text-light-muted-text dark:text-dark-muted-text">
                            Update the details for this video.
                        </p>
                    </div>
                </div>
                <div className="space-y-6">
                    <div>
                        <label htmlFor="filename" className="block text-base font-semibold text-light-text dark:text-dark-text mb-4">
                            Filename
                        </label>
                        <input
                            id="filename"
                            type="text"
                            placeholder="Enter a new filename"
                            value={filename}
                            onChange={(e) => setFilename(e.target.value)}
                            className="w-full px-6 py-4 bg-light-muted-background dark:bg-dark-muted-background border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-2xl text-light-text dark:text-dark-text focus:outline-none focus:ring-3 focus:ring-light-highlight/20 dark:focus:ring-dark-highlight/20 transition-all"
                            disabled={isSaving}
                        />
                    </div>
                    <div>
                        <label htmlFor="description" className="block text-base font-semibold text-light-text dark:text-dark-text mb-4">
                            Description
                        </label>
                        <textarea
                            id="description"
                            placeholder="Add a detailed description for your video analysis..."
                            value={description}
                            onChange={(e) => setDescription(e.target.value)}
                            className="w-full px-6 py-4 bg-light-muted-background dark:bg-dark-muted-background border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-2xl text-light-text dark:text-dark-text focus:outline-none focus:ring-3 focus:ring-light-highlight/20 dark:focus:ring-dark-highlight/20 transition-all resize-none"
                            rows={5}
                            disabled={isSaving}
                        />
                    </div>
                    <Button
                        onClick={handleSave}
                        disabled={isSaving}
                        className="w-full bg-gradient-to-r from-light-highlight via-light-highlight to-light-highlight/80 dark:from-dark-highlight dark:via-dark-highlight dark:to-dark-highlight/80 text-white font-bold py-5 px-8 rounded-2xl shadow-xl hover:shadow-2xl transition-all duration-300 hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 flex items-center justify-center space-x-4 text-lg"
                    >
                        {isSaving ? (
                            <>
                                <Loader2 className="w-6 h-6 animate-spin" />
                                <span>Saving...</span>
                            </>
                        ) : (
                            <>
                                <Save className="w-6 h-6" />
                                <span>Save Changes</span>
                            </>
                        )}
                    </Button>
                </div>
            </div>
        </div>
    );
};

// --- Delete Confirmation Modal Component ---
const DeleteVideoModal = ({ isOpen, onClose, video, onDeleteSuccess }) => {
    const [isDeleting, setIsDeleting] = useState(false);

    const getAuthToken = () => {
        return localStorage.getItem("authToken") || sessionStorage.getItem("authToken");
    };

    const handleDelete = async () => {
        if (!video) return;

        setIsDeleting(true);
        const loadingToast = showToast.loading("Deleting video...");

        try {
            const token = getAuthToken();
            const response = await fetch(API_ENDPOINTS.DELETE_VIDEO(video.id), {
                method: "DELETE",
                headers: {
                    Authorization: `Bearer ${token}`,
                },
            });

            const result = await response.json();
            showToast.dismiss(loadingToast);

            if (response.ok && result.success) {
                showToast.success("Video deleted successfully.");
                onDeleteSuccess();
            } else {
                showToast.error(result.message || "Failed to delete video.");
            }
        } catch (error) {
            showToast.dismiss(loadingToast);
            showToast.error("An error occurred while deleting the video.");
        } finally {
            setIsDeleting(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-light-background/80 dark:bg-dark-background/80 backdrop-blur-sm transition-opacity duration-300">
            <div className="relative w-full max-w-md max-h-[90vh] overflow-y-auto bg-light-background dark:bg-dark-background rounded-3xl p-8 shadow-2xl border border-red-500/10 dark:border-red-500/10">
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 p-2 rounded-full text-light-muted-text dark:text-dark-muted-text hover:bg-light-muted-background/50 dark:hover:bg-dark-muted-background/50 transition-colors"
                >
                    <X className="w-5 h-5" />
                </button>
                <div className="flex items-start space-x-4 mb-6">
                    <div className="p-3 bg-red-500/10 rounded-2xl">
                        <AlertTriangle className="w-8 h-8 text-red-500" />
                    </div>
                    <div>
                        <h2 className="text-2xl font-bold text-light-text dark:text-dark-text">
                            Confirm Deletion
                        </h2>
                        <p className="text-light-muted-text dark:text-dark-muted-text mt-1">
                            Are you sure you want to delete this video? This action cannot be undone.
                        </p>
                    </div>
                </div>
                <div className="text-light-text dark:text-dark-text font-semibold mb-6">
                    Video: {video?.filename}
                </div>
                <div className="flex justify-end space-x-4">
                    <Button onClick={onClose} variant="outline" disabled={isDeleting}>
                        Cancel
                    </Button>
                    <Button
                        onClick={handleDelete}
                        disabled={isDeleting}
                        className="bg-red-500 hover:bg-red-600 text-white font-bold transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {isDeleting ? (
                            <>
                                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                Deleting...
                            </>
                        ) : (
                            <>
                                <Trash2 className="w-4 h-4 mr-2" />
                                Delete
                            </>
                        )}
                    </Button>
                </div>
            </div>
        </div>
    );
};

// --- Main Dashboard Component ---
export const Dashboard = () => {
    const [videos, setVideos] = useState([]);
    const [loading, setLoading] = useState(true);
    const [analyzing, setAnalyzing] = useState({});
    const [analyticsDataLoading, setAnalyticsDataLoading] = useState(false);
    const [showUploadModal, setShowUploadModal] = useState(false);
    const [showEditModal, setShowEditModal] = useState(false);
    const [showDeleteModal, setShowDeleteModal] = useState(false);
    const [videoToEdit, setVideoToEdit] = useState(null);
    const [videoToDelete, setVideoToDelete] = useState(null);
    const navigate = useNavigate();

    const getAuthToken = useCallback(() => localStorage.getItem("authToken") || sessionStorage.getItem("authToken"), []);

    const fetchVideos = useCallback(async (isInitialLoad = false) => {
        if (isInitialLoad) setLoading(true);
        try {
            const token = getAuthToken();
            // Using API_ENDPOINTS.GET_VIDEOS for now. Once the API supports it, use API_ENDPOINTS.GET_ANALYSES
            const response = await fetch(API_ENDPOINTS.GET_VIDEOS, { headers: { Authorization: `Bearer ${token}` } });
            const data = await response.json();
            
            if (data.success) {
                setVideos(data.data);
                const initialModels = data.data.reduce((acc, video) => {
                    acc[video.id] = ANALYSIS_MODELS.find(m => !video.analyses.some(a => a.model === m)) || "";
                    return acc;
                }, {});
            } else {
                showToast.error("Failed to load videos.");
            }
        } catch (error) {
            showToast.error("An error occurred while fetching videos.");
        } finally {
            if (isInitialLoad) setLoading(false);
        }
    }, [getAuthToken]);

    const handleRefresh = useCallback(() => {
        setAnalyticsDataLoading(true);
        fetchVideos()
            .finally(() => setAnalyticsDataLoading(false));
        showToast.success("Data Refreshed Successfully!")
    }, [fetchVideos]);

    useEffect(() => {
        fetchVideos(true);
    }, [fetchVideos]);

    const analyzeVideo = useCallback(async (videoId, model) => {
        if (!model) {
            return showToast.error("Please select a model.");
        }
        setAnalyzing(prev => ({ ...prev, [videoId]: true }));
        try {
            const token = getAuthToken();
            const response = await fetch(API_ENDPOINTS.ANALYZE_VIDEO(videoId), {
                method: "POST",
                headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
                body: JSON.stringify({ model }),
            });
            const result = await response.json();
            if (result.success) {
                showToast.success(`Analysis started for ${model}.`);
            } else {
                showToast.error(result.message || "Analysis request failed.");
            }
            fetchVideos();
        } catch (error) {
            showToast.error("Analysis request failed.");
        } finally {
            setAnalyzing(prev => ({ ...prev, [videoId]: false }));
        }
    }, [getAuthToken, fetchVideos]);

    const handleEditClick = (event, video) => {
        event.stopPropagation();
        setVideoToEdit(video);
        setShowEditModal(true);
    };

    const handleDeleteClick = (event, video) => {
        event.stopPropagation();
        setVideoToDelete(video);
        setShowDeleteModal(true);
    };

    const handleEditSuccess = () => {
        setShowEditModal(false);
        setVideoToEdit(null);
        fetchVideos();
    };

    const handleDeleteSuccess = () => {
        setShowDeleteModal(false);
        setVideoToDelete(null);
        fetchVideos();
    };


    // Calculate stats from the fetched video data
    const stats = useMemo(() => {
        return videos.reduce((acc, v) => {
            acc.total++;
            if (v.status === 'ANALYZED') acc.analyzed++;
            v.analyses.forEach(a => {
                if (a.prediction === 'REAL') acc.realDetections++;
                if (a.prediction === 'FAKE') acc.fakeDetections++;
            });
            return acc;
        }, { total: 0, analyzed: 0, realDetections: 0, fakeDetections: 0 });
    }, [videos]);

    // Define columns for the DataTable.
    const columns = useMemo(() => [
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
            sortable: false, 
            filterable: false,
            accessor: (item) => item.description || "N/A",
        },
        { 
            key: "size",
            header: "File Size",
            sortable: true,
            accessor: (item) => formatBytes(item.size),
        },
        { 
            key: "createdAt",
            header: "Upload Date",
            sortable: true,
            accessor: (item) => formatDate(item.createdAt),
        },
        { key: "status", header: "Status", sortable: true, accessor: (item) => <StatusBadge status={item.status} /> },
        { key: "analyses", header: "Analyses", accessor: (item) => <ProgressIndicator analyses={item.analyses} /> },
        {
            key: "actions",
            header: "Actions",
            align: "center",
            accessor: (item) => {
                const isAnalyzed = item.status === 'ANALYZED';
                const isProcessing = analyzing[item.id];
                const availableModels = ANALYSIS_MODELS.filter(model => !item.analyses.some(a => a.model === model));

                return (
                    <div className="flex items-center justify-center space-x-2">
                        {isAnalyzed ? (
                            <Button
                                variant="outline"
                                size="sm"
                                onClick={(e) => { e.stopPropagation(); navigate(`/results/${item.id}`); }}
                            >
                                View Results
                            </Button>
                        ) : isProcessing ? (
                            <Button variant="outline" size="sm" disabled>
                                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                Analyzing...
                            </Button>
                        ) : (
                            <DropdownMenu>
                                <DropdownMenuTrigger asChild>
                                    <Button variant="outline" size="icon" onClick={(e) => e.stopPropagation()}>
                                        <MoreHorizontal className="h-4 w-4" />
                                    </Button>
                                </DropdownMenuTrigger>
                                <DropdownMenuContent align="end">
                                    {availableModels.length > 0 && availableModels.map(model => (
                                        <DropdownMenuItem key={model} onClick={() => analyzeVideo(item.id, model)}>
                                            Analyze with {model}
                                        </DropdownMenuItem>
                                    ))}
                                    {availableModels.length === 0 && (
                                        <DropdownMenuItem disabled>
                                            No available models
                                        </DropdownMenuItem>
                                    )}
                                </DropdownMenuContent>
                            </DropdownMenu>
                        )}
                        <Button
                            variant="ghost"
                            size="icon"
                            onClick={(e) => handleEditClick(e, item)}
                        >
                            <Edit className="h-4 w-4 text-light-highlight dark:text-dark-highlight"/>
                        </Button>
                        <Button
                            variant="ghost"
                            size="icon"
                            onClick={(e) => handleDeleteClick(e, item)}
                        >
                            <Trash2 className="h-4 w-4 text-red-500"/>
                        </Button>
                    </div>
                );
            },
        },
    ], [analyzing, navigate, analyzeVideo, handleEditClick, handleDeleteClick]);


    if (loading) {
        return <PageLoader text="Loading Dashboard..."/>;
    }
    
    return (
        <div className="w-full min-h-screen space-y-6">
            <Card className="bg-light-background dark:bg-dark-muted-background p-6 rounded-xl shadow-sm border border-light-secondary dark:border-dark-secondary">
                <div className="transition-colors duration-200 relative z-10 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                    <div>
                        <h1 className="text-3xl md:text-4xl font-extrabold text-light-text dark:text-dark-text flex items-center gap-3">
                            Drishtiksha Dashboard
                        </h1>
                        <p className="text-base text-light-muted-text dark:text-dark-muted-text flex items-center gap-2 mt-2">
                            Comprehensive deepfake detection service
                        </p>
                    </div>

                    <div className="flex items-center gap-2">
                        <Button
                            onClick={handleRefresh}
                            disabled={analyticsDataLoading}
                            variant="outline"
                            className="flex items-center rounded-xl gap-1.5 p-3 py-5"
                        >
                            <RefreshCw className={`w-5 h-5 text-light-highlight flex items-center justify-center ${analyticsDataLoading ? "animate-spin" : ""}`} />
                            {analyticsDataLoading ? "Refreshing ..." : "Refresh Data"}
                        </Button>
                        <Button
                            onClick={() => setShowUploadModal(true)}
                            disabled={analyticsDataLoading}
                            variant="outline"
                            className="flex items-center rounded-xl gap-1.5 p-3 py-5"
                        >
                            <UploadCloud className={`w-5 h-5 text-light-highlight flex items-center justify-center `} />
                            Upload Video
                        </Button>
                    </div>
                </div>
            </Card>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-6">
                <StatCard title="Total Videos" value={stats.total} icon={Play} onClick={()=>{}} />
                <StatCard title="Videos Analyzed" value={stats.analyzed} icon={CheckCircle} onClick={()=>{}} />
                <StatCard title="Real Videos Detected" value={stats.realDetections} icon={AlertTriangle} onClick={()=>{}} />
                <StatCard title="Deepfake Videos Detected" value={stats.fakeDetections} icon={AlertTriangle} onClick={()=>{}} />
            </div>
            {/* Main DataTable Component */}
            <DataTable
                columns={columns}
                data={videos}
                pageSize={10}
                onRowClick={(item) => navigate(`/results/${item.id}`)}
                showCard={true}
                showSearch={true}
                showPagination={true}
                emptyMessage="No videos found. Upload one to get started!"
                searchPlaceholder="Search by filename..."
            />

            {/* Render the Upload Modal */}
            <UploadModal 
                isOpen={showUploadModal} 
                onClose={() => setShowUploadModal(false)}
                onUploadSuccess={() => {
                    setShowUploadModal(false);
                    fetchVideos();
                }}
            />

            {/* Render the Edit Video Modal */}
            {videoToEdit && (
                <EditVideoModal
                    isOpen={showEditModal}
                    onClose={() => {
                        setShowEditModal(false);
                        setVideoToEdit(null);
                    }}
                    video={videoToEdit}
                    onUpdateSuccess={handleEditSuccess}
                />
            )}

            {/* Render the Delete Confirmation Modal */}
            {videoToDelete && (
                <DeleteVideoModal
                    isOpen={showDeleteModal}
                    onClose={() => {
                        setShowDeleteModal(false);
                        setVideoToDelete(null);
                    }}
                    video={videoToDelete}
                    onDeleteSuccess={handleDeleteSuccess}
                />
            )}
        </div>
    );
};

export default Dashboard;
