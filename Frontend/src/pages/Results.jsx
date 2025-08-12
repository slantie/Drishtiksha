/**
 * @file src/pages/results.jsx
 * @description The results page displays the video details, a video player, and the deepfake analysis results.
 */

import React, { useState, useEffect, useCallback, useRef } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import ReactPlayer from "react-player";
import {
    Loader2, AlertTriangle, ArrowLeft, Cpu, ShieldCheck, ShieldAlert,
    Play, Pause, Volume2, VolumeX, RotateCcw, FileVideo, RefreshCw,
    Edit, Trash, Save, X, Trash2
} from "lucide-react";
import { showToast } from "../utils/toast";
import { API_ENDPOINTS } from "../constants/apiEndpoints";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";

//==================================================================
// Helper functions for formatting data and Cloudinary transformations
//==================================================================
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
 * Transforms a raw Cloudinary video URL to ensure broad browser compatibility.
 * It inserts auto-quality (q_auto:good) and auto-codec (vc_auto) parameters.
 * @param {string} url - The original Cloudinary video URL.
 * @returns {string} - The transformed URL.
 */
const getTransformedVideoUrl = (url) => {
    if (!url) return null;
    // We split the URL and inject the transformation parameters before the version number.
    const parts = url.split('/upload/');
    if (parts.length === 2) {
        return `${parts[0]}/upload/q_auto:good,vc_auto/${parts[1]}`;
    }
    return url;
};

//==================================================================
// Header Component
//==================================================================
const ResultsHeader = ({ filename, description, onRefresh, onEditClick, onDeleteClick }) => (
    <>
    <div className="bg-light-background dark:bg-dark-muted-background p-6 rounded-xl shadow-sm border border-light-secondary dark:border-dark-secondary">
        <div className="transition-colors duration-200 relative z-10 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div>
                <h1 className="text-3xl md:text-4xl font-extrabold text-light-text dark:text-dark-text flex items-center gap-3">
                    Video Analysis Results
                </h1>
                <p className="text-base text-light-muted-text dark:text-dark-muted-text flex items-center gap-2 mt-2">
                    Filename: <span className="text-light-highlight dark:text-dark-highlight">{filename}</span>
                </p>
            </div>

            <div className="flex items-center gap-2">
                <Button
                    onClick={onRefresh}
                    variant="outline"
                    className="flex items-center rounded-xl gap-1.5 p-3 py-5"
                >
                    <RefreshCw className={`w-5 h-5 text-light-highlight flex items-center justify-center`} />
                    Refresh Data
                </Button>
                <Button
                    onClick={onEditClick}
                    variant="outline"
                    className="flex items-center rounded-xl gap-1.5 p-3 py-5"
                >
                    <Edit className={`w-5 h-5 text-light-highlight flex items-center justify-center `} />
                    {/* Edit Video Details */}
                </Button>
                <Button
                    onClick={onDeleteClick}
                    variant="destructive"
                    className="flex items-center rounded-xl gap-1.5 p-3 py-5"
                >
                    <Trash className={`w-5 h-5 text-red-500 flex items-center justify-center `} />
                    {/* Delete Video */}
                </Button>
            </div>
        </div>
    </div>
    </>
);

//==================================================================
// Video Details & Player Component
//==================================================================
const VideoDetails = ({ video }) => {
    const [isPlaying, setIsPlaying] = useState(false);
    const [isMuted, setIsMuted] = useState(false);
    const [volume, setVolume] = useState(0.8);
    const playerRef = useRef(null);

    const handlePlayPause = () => setIsPlaying(!isPlaying);
    const handleMute = () => setIsMuted(!isMuted);

    // Get the transformed URL for playback
    const transformedUrl = getTransformedVideoUrl(video.url);

    return (
        <div className="bg-light-background dark:bg-dark-muted-background rounded-xl shadow-md border border-light-secondary dark:border-dark-secondary p-6 space-y-6">
            <div className="relative w-full aspect-video rounded-lg overflow-hidden bg-black">
                {transformedUrl ? (
                    <ReactPlayer
                        ref={playerRef}
                        // Use the transformed URL here
                        url={transformedUrl}
                        playing={isPlaying}
                        muted={isMuted}
                        volume={isMuted ? 0 : volume}
                        controls={false}
                        width="100%"
                        height="100%"
                        onEnded={() => setIsPlaying(false)}
                    />
                ) : (
                    <div className="absolute inset-0 flex items-center justify-center bg-light-muted-background dark:bg-dark-background/50">
                        <FileVideo className="w-16 h-16 text-light-muted-text/30 dark:text-dark-muted-text/30" />
                    </div>
                )}
                {/* Custom Controls Overlay */}
                <div className="absolute inset-0 flex items-center justify-center transition-opacity duration-300 bg-black/50 opacity-0 hover:opacity-100">
                    <button
                        onClick={handlePlayPause}
                        className="p-4 rounded-full bg-white/20 hover:bg-white/40 text-white backdrop-blur-sm transition-colors"
                    >
                        {isPlaying ? <Pause className="w-8 h-8" /> : <Play className="w-8 h-8" />}
                    </button>
                    <div className="absolute bottom-4 left-4 flex items-center space-x-2">
                        <button
                            onClick={handleMute}
                            className="p-2 rounded-full text-white bg-white/20 hover:bg-white/40 transition-colors"
                        >
                            {isMuted || volume === 0 ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
                        </button>
                        <input
                            type="range"
                            min={0}
                            max={1}
                            step="any"
                            value={volume}
                            onChange={(e) => {
                                setVolume(parseFloat(e.target.value));
                                setIsMuted(false);
                            }}
                            className="w-20 h-1 appearance-none bg-white/50 rounded-full [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-light-highlight [&::-webkit-slider-thumb]:appearance-none"
                        />
                    </div>
                </div>
            </div>
            
            <div className="space-y-4">
                <h3 className="text-lg font-bold text-light-text dark:text-dark-text">Video Information</h3>
                <div className="space-y-3 text-sm">
                    <div className="flex justify-between">
                        <span className="font-medium text-light-muted-text dark:text-dark-muted-text">File Size:</span>
                        <span className="font-mono text-light-text dark:text-dark-text">{formatBytes(video.size)}</span>
                    </div>
                    <div className="flex justify-between">
                        <span className="font-medium text-light-muted-text dark:text-dark-muted-text">MIME Type:</span>
                        <span className="font-mono text-light-text dark:text-dark-text">{video.mimetype}</span>
                    </div>
                    <div className="flex justify-between">
                        <span className="font-medium text-light-muted-text dark:text-dark-muted-text">Date Uploaded:</span>
                        <span className="font-mono text-light-text dark:text-dark-text">{new Date(video.createdAt).toLocaleString()}</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

//==================================================================
// Analysis Result Card Component
//==================================================================
const AnalysisCard = ({ analysis }) => {
    const isReal = analysis.prediction === "REAL";
    const confidence = (analysis.confidence * 100);

    const cardStyles = isReal
        ? "bg-green-500/5 dark:bg-green-500/10 border-green-500/30"
        : "bg-red-500/5 dark:bg-red-500/10 border-red-500/30";
    
    const Icon = isReal ? ShieldCheck : ShieldAlert;
    const iconColor = isReal ? "text-green-500" : "text-red-500";
    const confidenceColor = isReal ? "text-green-500" : "text-red-500";
    const statusText = isReal ? "Likely Real" : "Likely Deepfake";

    return (
        <div className={`rounded-3xl border-2 ${cardStyles} p-8 space-y-6 transition-all hover:shadow-xl`}>
            {/* Header: Model & Prediction Icon */}
            <div className="flex justify-between items-center">
                <div className="flex items-center gap-4">
                    <div className="p-3 bg-light-highlight/10 dark:bg-dark-highlight/10 rounded-xl">
                        <Cpu className="w-6 h-6 text-light-highlight dark:text-dark-highlight" />
                    </div>
                    <h3 className="text-2xl font-bold text-light-text dark:text-dark-text">{analysis.model}</h3>
                </div>
                <div className={`p-2 rounded-full ${iconColor} bg-white dark:bg-dark-muted-background border border-light-muted-text/10 dark:border-dark-muted-text/10 shadow-lg`}>
                    <Icon className="w-8 h-8" />
                </div>
            </div>

            {/* Confidence Score and Progress Bar */}
            <div className="space-y-4">
                <div className="flex justify-between items-end">
                    <p className={`text-4xl font-extrabold ${confidenceColor}`}>{confidence.toFixed(1)}<span className="text-2xl font-semibold">%</span></p>
                    <p className={`text-base font-semibold ${confidenceColor}`}>{statusText}</p>
                </div>
                <div className="w-full bg-light-secondary dark:bg-dark-secondary rounded-full h-2.5">
                    <div
                        className={`${isReal ? 'bg-green-500' : 'bg-red-500'} h-2.5 rounded-full`}
                        style={{ width: `${confidence}%` }}
                    />
                </div>
            </div>

            {/* Metadata Table */}
            <div className="border-t border-light-muted-text/10 dark:border-dark-muted-text/10 pt-6 text-sm space-y-3">
                <div className="flex justify-between">
                    <span className="font-medium text-light-muted-text dark:text-dark-muted-text">Processing Time:</span>
                    <span className="font-mono text-light-text dark:text-dark-text">{analysis.processingTime?.toFixed(2)}s</span>
                </div>
                <div className="flex justify-between">
                    <span className="font-medium text-light-muted-text dark:text-dark-muted-text">Analyzed On:</span>
                    <span className="font-mono text-light-text dark:text-dark-text">{new Date(analysis.createdAt).toLocaleDateString()}</span>
                </div>
            </div>
        </div>
    );
};

//==================================================================
// Edit Video Modal Component
//==================================================================
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

//==================================================================
// Delete Confirmation Modal Component
//==================================================================
const DeleteVideoModal = ({ isOpen, onClose, video, onDeleteSuccess }) => {
    const [isDeleting, setIsDeleting] = useState(false);
    const navigate = useNavigate();

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
                navigate('/dashboard'); // Navigate back to the dashboard after successful deletion
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


//==================================================================
// Main Results Page Component
//==================================================================
export const Results = () => {
    const { videoId } = useParams();
    const [videoData, setVideoData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [showEditModal, setShowEditModal] = useState(false);
    const [showDeleteModal, setShowDeleteModal] = useState(false);

    const getAuthToken = useCallback(() => localStorage.getItem("authToken") || sessionStorage.getItem("authToken"), []);

    const fetchVideoDetails = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const token = getAuthToken();
            const response = await fetch(`${API_ENDPOINTS.GET_VIDEOS}/${videoId}`, {
                headers: { Authorization: `Bearer ${token}` },
            });
            const data = await response.json();
            if (data.success) {
                setVideoData(data.data);
            } else {
                throw new Error(data.message || "Failed to load video details.");
            }
        } catch (err) {
            setError(err.message);
            showToast.error(err.message);
        } finally {
            setLoading(false);
        }
    }, [videoId, getAuthToken]);

    useEffect(() => {
        fetchVideoDetails();
    }, [fetchVideoDetails]);

    if (loading) {
        return (
            <div className="min-h-screen flex flex-col items-center justify-center p-8">
                <Loader2 className="w-16 h-16 animate-spin text-light-highlight dark:text-dark-highlight mb-4" />
                <p className="text-xl font-medium text-light-muted-text dark:text-dark-muted-text">Loading video analysis...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex flex-col items-center justify-center min-h-screen p-8">
                <div className="text-center p-12 max-w-lg mx-auto bg-light-background dark:bg-dark-muted-background rounded-3xl shadow-xl border border-red-500/10">
                    <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-6" />
                    <h2 className="text-2xl font-bold text-light-text dark:text-dark-text">An Error Occurred</h2>
                    <p className="text-light-muted-text dark:text-dark-muted-text mt-2 mb-6">{error}</p>
                    <Link to="/dashboard">
                        <Button className="inline-flex items-center gap-2 bg-light-highlight dark:bg-dark-highlight text-white font-semibold py-4 px-6 rounded-xl hover:bg-light-highlight/90 dark:hover:bg-dark-highlight/90 transition-colors">
                            <ArrowLeft className="w-5 h-5" />
                            <span>Go Back to Dashboard</span>
                        </Button>
                    </Link>
                </div>
            </div>
        );
    }

    if (!videoData) return null;

    return (
        <div className="w-full mx-auto space-y-8">
            <ResultsHeader
                filename={videoData.filename}
                description={videoData.description}
                onRefresh={fetchVideoDetails}
                onEditClick={() => setShowEditModal(true)}
                onDeleteClick={() => setShowDeleteModal(true)}
            />

            <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
                {/* Left Column: Video Player & Details */}
                <div className="xl:col-span-1 space-y-8">
                    <VideoDetails video={videoData} />
                </div>

                {/* Right Column: Analysis Results */}
                <div className="xl:col-span-2 space-y-6">
                    {videoData.analyses && videoData.analyses.length > 0 ? (
                        videoData.analyses
                            .sort((a, b) => b.confidence - a.confidence)
                            .map((analysis) => (
                                <AnalysisCard key={analysis.id} analysis={analysis} />
                            ))
                    ) : (
                        <div className="h-full flex flex-col items-center justify-center bg-light-background dark:bg-dark-muted-background rounded-3xl shadow-md border border-light-secondary dark:border-dark-secondary p-12 text-center">
                            <RotateCcw className="w-16 h-16 text-light-muted-text/30 dark:text-dark-muted-text/30 animate-pulse mb-4" />
                            <h2 className="text-xl font-bold text-light-text dark:text-dark-text">No Analyses Found</h2>
                            <p className="mt-2 text-light-muted-text dark:text-dark-muted-text">
                                This video has not been analyzed yet. Return to the dashboard to select a model.
                            </p>
                        </div>
                    )}
                </div>
            </div>

            {/* Modals */}
            {showEditModal && <EditVideoModal
                isOpen={showEditModal}
                onClose={() => setShowEditModal(false)}
                video={videoData}
                onUpdateSuccess={fetchVideoDetails}
            />}
            {showDeleteModal && <DeleteVideoModal
                isOpen={showDeleteModal}
                onClose={() => setShowDeleteModal(false)}
                video={videoData}
                onDeleteSuccess={() => {
                    // Navigate away after deletion
                    setShowDeleteModal(false);
                    // The useNavigate hook is defined inside the DeleteVideoModal
                    // The navigate call is handled there
                }}
            />}
        </div>
    );
};

export default Results;
