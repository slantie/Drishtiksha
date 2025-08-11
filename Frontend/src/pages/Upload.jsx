import React, { useState, useRef } from "react";
import ReactPlayer from "react-player";
import {
    Upload,
    File,
    Play,
    Pause,
    Volume2,
    VolumeX,
    RotateCcw,
    CheckCircle,
    AlertCircle,
    FileVideo,
    Info,
    Maximize2,
    Settings,
} from "lucide-react";
import { showToast } from "../utils/toast";

const backendUrl = import.meta.env.VITE_BACKEND_URL || "http://localhost:4000";

export const UploadPage = () => {
    const [videoFile, setVideoFile] = useState(null);
    const [description, setDescription] = useState("");
    const [isUploading, setIsUploading] = useState(false);
    const [videoPreviewUrl, setVideoPreviewUrl] = useState(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [isMuted, setIsMuted] = useState(false);
    const [dragActive, setDragActive] = useState(false);
    const [volume, setVolume] = useState(0.8);
    const [duration, setDuration] = useState(0);
    const [playedSeconds, setPlayedSeconds] = useState(0);
    const playerRef = useRef(null);
    const fileInputRef = useRef(null);

    const getAuthToken = () => {
        return (
            localStorage.getItem("authToken") ||
            sessionStorage.getItem("authToken")
        );
    };

    const handleFileSelect = (file) => {
        if (file && file.type.startsWith("video/")) {
            setVideoFile(file);

            // Create preview URL
            if (videoPreviewUrl) {
                URL.revokeObjectURL(videoPreviewUrl);
            }

            const previewUrl = URL.createObjectURL(file);
            setVideoPreviewUrl(previewUrl);
            setIsPlaying(false);
            setPlayedSeconds(0);
        } else {
            showToast.error("Please select a valid video file.");
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

    const togglePlayPause = () => {
        setIsPlaying(!isPlaying);
    };

    const toggleMute = () => {
        setIsMuted(!isMuted);
    };

    const handleSeek = (seconds) => {
        if (playerRef.current) {
            playerRef.current.seekTo(seconds);
            setPlayedSeconds(seconds);
        }
    };

    const clearVideo = () => {
        setVideoFile(null);
        if (videoPreviewUrl) {
            URL.revokeObjectURL(videoPreviewUrl);
        }
        setVideoPreviewUrl(null);
        setIsPlaying(false);
        setPlayedSeconds(0);
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    };

    const formatFileSize = (bytes) => {
        if (bytes === 0) return "0 Bytes";
        const k = 1024;
        const sizes = ["Bytes", "KB", "MB", "GB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
    };

    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, "0")}`;
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
            console.log("Uploading to:", `${backendUrl}/api/video/upload`);

            const res = await fetch(`${backendUrl}/api/video/upload`, {
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
                    "Video uploaded successfully! You can now analyze it."
                );

                // Clear the form
                clearVideo();
                setDescription("");

                // Redirect to dashboard after a short delay
                setTimeout(() => {
                    window.location.href = "/dashboard";
                }, 2000);
            } else {
                showToast.error(
                    data.message || "Upload failed. Please try again."
                );
            }
        } catch (err) {
            showToast.dismiss(loadingToast);
            console.error("Upload error:", err);

            if (err.name === "TypeError" && err.message.includes("fetch")) {
                showToast.error(
                    "Cannot connect to server. Please check if the backend is running on port 4000."
                );
            } else {
                showToast.error(
                    "An error occurred while uploading. Please try again."
                );
            }
        } finally {
            setIsUploading(false);
        }
    };

    // Cleanup preview URL on unmount
    React.useEffect(() => {
        return () => {
            if (videoPreviewUrl) {
                URL.revokeObjectURL(videoPreviewUrl);
            }
        };
    }, [videoPreviewUrl]);

    return (
        <div className="min-h-screen bg-gradient-to-br from-light-background via-light-muted-background to-light-background dark:from-dark-background dark:via-dark-muted-background dark:to-dark-background">
            {/* Full Width Container */}
            <div className="w-full px-6 py-8">
                {/* Main Content - Simplified and More Effective Layout */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 w-full max-w-[1800px] mx-auto">
                    {/* Left Column - Upload Form (Takes 1/3 on lg and up) */}
                    <div className="lg:col-span-1 space-y-8">
                        {/* Upload Form Card */}
                        <div className="bg-light-background dark:bg-dark-background rounded-3xl p-8 shadow-2xl border border-light-muted-text/10 dark:border-dark-muted-text/10">
                            <div className="flex items-center space-x-4 mb-8">
                                <div className="p-3 bg-light-highlight/10 dark:bg-dark-highlight/10 rounded-2xl">
                                    <Upload className="w-8 h-8 text-light-highlight dark:text-dark-highlight" />
                                </div>
                                <div>
                                    <h2 className="text-2xl font-bold text-light-text dark:text-dark-text">
                                        Upload Details
                                    </h2>
                                    <p className="text-light-muted-text dark:text-dark-muted-text">
                                        Select and configure your video
                                    </p>
                                </div>
                            </div>

                            <form onSubmit={handleSubmit} className="space-y-8">
                                {/* Enhanced File Upload Area */}
                                <div>
                                    <label className="block text-base font-semibold text-light-text dark:text-dark-text mb-4">
                                        Select Video File
                                    </label>

                                    <div
                                        className={`relative border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-300 cursor-pointer ${
                                            dragActive
                                                ? "border-light-highlight dark:border-dark-highlight bg-light-highlight/5 dark:bg-dark-highlight/5 scale-105"
                                                : "border-light-muted-text/30 dark:border-dark-muted-text/30 hover:border-light-highlight/50 dark:hover:border-dark-highlight/50 hover:bg-light-highlight/2 dark:hover:bg-dark-highlight/2"
                                        }`}
                                        onDrop={handleDrop}
                                        onDragOver={handleDragOver}
                                        onDragLeave={handleDragLeave}
                                        onClick={() =>
                                            fileInputRef.current?.click()
                                        }
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
                                                        {formatFileSize(
                                                            videoFile.size
                                                        )}{" "}
                                                        • {videoFile.type}
                                                    </p>
                                                    <p className="text-sm text-light-muted-text dark:text-dark-muted-text">
                                                        Modified:{" "}
                                                        {new Date(
                                                            videoFile.lastModified
                                                        ).toLocaleDateString()}
                                                    </p>
                                                </div>
                                                <button
                                                    type="button"
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        clearVideo();
                                                    }}
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
                                                        Drag & drop your video
                                                        here
                                                    </p>
                                                    <p className="text-base text-light-muted-text dark:text-dark-muted-text mt-2">
                                                        or click to browse files
                                                    </p>
                                                </div>
                                                <div className="bg-light-muted-background/50 dark:bg-dark-muted-background/50 rounded-lg p-4 mt-4">
                                                    <p className="text-sm text-light-muted-text dark:text-dark-muted-text">
                                                        <strong>
                                                            Supported formats:
                                                        </strong>{" "}
                                                        MP4, AVI, MOV, WMV
                                                        <br />
                                                        <strong>
                                                            Maximum size:
                                                        </strong>{" "}
                                                        100MB
                                                        <br />
                                                        <strong>
                                                            Recommended:
                                                        </strong>{" "}
                                                        High quality videos for
                                                        better analysis
                                                    </p>
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </div>

                                {/* Enhanced Description */}
                                <div>
                                    <label className="block text-base font-semibold text-light-text dark:text-dark-text mb-4">
                                        Description (Optional)
                                    </label>
                                    <textarea
                                        placeholder="Add a detailed description for your video analysis. This helps in organizing and identifying your uploads..."
                                        value={description}
                                        onChange={(e) =>
                                            setDescription(e.target.value)
                                        }
                                        className="w-full px-6 py-4 bg-light-muted-background dark:bg-dark-muted-background border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-2xl text-light-text dark:text-dark-text focus:outline-none focus:ring-3 focus:ring-light-highlight/20 dark:focus:ring-dark-highlight/20 focus:border-light-highlight dark:focus:border-dark-highlight transition-all resize-none"
                                        rows={5}
                                        disabled={isUploading}
                                    />
                                </div>

                                {/* Enhanced Upload Button */}
                                <button
                                    type="submit"
                                    disabled={isUploading || !videoFile}
                                    className="w-full bg-gradient-to-r from-light-highlight via-light-highlight to-light-highlight/80 dark:from-dark-highlight dark:via-dark-highlight dark:to-dark-highlight/80 text-white font-bold py-5 px-8 rounded-2xl shadow-xl hover:shadow-2xl transition-all duration-300 hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 flex items-center justify-center space-x-4 text-lg"
                                >
                                    {isUploading ? (
                                        <>
                                            <div className="w-6 h-6 border-3 border-white/30 border-t-white rounded-full animate-spin"></div>
                                            <span>
                                                Uploading & Processing...
                                            </span>
                                        </>
                                    ) : (
                                        <>
                                            <Upload className="w-6 h-6" />
                                            <span>Upload & Start Analysis</span>
                                        </>
                                    )}
                                </button>
                            </form>
                        </div>
                    </div>

                    {/* Right Column - Video Preview (Takes 2/3 on lg and up) */}
                    <div className="lg:col-span-2 space-y-8">
                        <div className="bg-light-background dark:bg-dark-background rounded-3xl p-8 shadow-2xl border border-light-muted-text/10 dark:border-dark-muted-text/10">
                            <div className="flex items-center justify-between mb-8">
                                <div className="flex items-center space-x-4">
                                    <div className="p-3 bg-purple-500/10 rounded-2xl">
                                        <Play className="w-8 h-8 text-purple-500" />
                                    </div>
                                    <div>
                                        <h2 className="text-2xl font-bold text-light-text dark:text-dark-text">
                                            Video Preview
                                        </h2>
                                        <p className="text-light-muted-text dark:text-dark-muted-text">
                                            {videoFile
                                                ? `${videoFile.name}`
                                                : "No video selected"}
                                        </p>
                                    </div>
                                </div>
                                {videoFile && (
                                    <div className="flex items-center space-x-2">
                                        <Maximize2 className="w-5 h-5 text-light-muted-text dark:text-dark-muted-text" />
                                        <span className="text-sm text-light-muted-text dark:text-dark-muted-text">
                                            {formatFileSize(videoFile.size)}
                                        </span>
                                    </div>
                                )}
                            </div>

                            {/* Enhanced Video Player */}
                            <div className="aspect-video bg-gradient-to-br from-light-muted-background to-light-muted-background/50 dark:from-dark-muted-background dark:to-dark-muted-background/50 rounded-2xl overflow-hidden border border-light-muted-text/10 dark:border-dark-muted-text/10 shadow-inner">
                                {videoPreviewUrl ? (
                                    <div className="relative w-full h-full">
                                        <ReactPlayer
                                            ref={playerRef}
                                            url={videoPreviewUrl}
                                            width="100%"
                                            height="100%"
                                            playing={isPlaying}
                                            muted={isMuted}
                                            volume={volume}
                                            controls={false}
                                            onDuration={setDuration}
                                            onProgress={({ playedSeconds }) =>
                                                setPlayedSeconds(playedSeconds)
                                            }
                                            style={{
                                                borderRadius: "16px",
                                                overflow: "hidden",
                                            }}
                                            config={{
                                                file: {
                                                    attributes: {
                                                        style: {
                                                            width: "100%",
                                                            height: "100%",
                                                            objectFit:
                                                                "contain",
                                                        },
                                                    },
                                                },
                                            }}
                                        />

                                        {/* Custom Video Controls */}
                                        <div className="absolute bottom-6 left-6 right-6">
                                            <div className="bg-black/80 backdrop-blur-lg rounded-2xl p-6 shadow-2xl">
                                                {/* Progress Bar */}
                                                <div className="mb-4">
                                                    <div className="flex items-center justify-between text-white/80 text-sm mb-2">
                                                        <span>
                                                            {formatTime(
                                                                playedSeconds
                                                            )}
                                                        </span>
                                                        <span>
                                                            {formatTime(
                                                                duration
                                                            )}
                                                        </span>
                                                    </div>
                                                    <div
                                                        className="w-full h-2 bg-white/20 rounded-full cursor-pointer"
                                                        onClick={(e) => {
                                                            const rect =
                                                                e.currentTarget.getBoundingClientRect();
                                                            const clickX =
                                                                e.clientX -
                                                                rect.left;
                                                            const newTime =
                                                                (clickX /
                                                                    rect.width) *
                                                                duration;
                                                            handleSeek(newTime);
                                                        }}
                                                    >
                                                        <div
                                                            className="h-full bg-gradient-to-r from-light-highlight to-light-highlight/80 rounded-full transition-all"
                                                            style={{
                                                                width: `${
                                                                    (playedSeconds /
                                                                        duration) *
                                                                    100
                                                                }%`,
                                                            }}
                                                        />
                                                    </div>
                                                </div>

                                                {/* Control Buttons */}
                                                <div className="flex items-center justify-between">
                                                    <div className="flex items-center space-x-4">
                                                        <button
                                                            onClick={
                                                                togglePlayPause
                                                            }
                                                            className="p-3 bg-light-highlight hover:bg-light-highlight/80 rounded-xl transition-colors shadow-lg"
                                                        >
                                                            {isPlaying ? (
                                                                <Pause className="w-6 h-6 text-white" />
                                                            ) : (
                                                                <Play className="w-6 h-6 text-white" />
                                                            )}
                                                        </button>

                                                        <button
                                                            onClick={toggleMute}
                                                            className="p-3 bg-white/20 hover:bg-white/30 rounded-xl transition-colors"
                                                        >
                                                            {isMuted ? (
                                                                <VolumeX className="w-5 h-5 text-white" />
                                                            ) : (
                                                                <Volume2 className="w-5 h-5 text-white" />
                                                            )}
                                                        </button>

                                                        <button
                                                            onClick={() =>
                                                                handleSeek(0)
                                                            }
                                                            className="p-3 bg-white/20 hover:bg-white/30 rounded-xl transition-colors"
                                                        >
                                                            <RotateCcw className="w-5 h-5 text-white" />
                                                        </button>
                                                    </div>

                                                    {/* Volume Control */}
                                                    <div className="flex items-center space-x-3">
                                                        <Volume2 className="w-4 h-4 text-white/60" />
                                                        <input
                                                            type="range"
                                                            min={0}
                                                            max={1}
                                                            step={0.1}
                                                            value={volume}
                                                            onChange={(e) =>
                                                                setVolume(
                                                                    parseFloat(
                                                                        e.target
                                                                            .value
                                                                    )
                                                                )
                                                            }
                                                            className="w-20 h-2 bg-white/20 rounded-lg appearance-none cursor-pointer"
                                                        />
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="w-full h-full flex items-center justify-center">
                                        <div className="text-center space-y-6">
                                            <div className="w-24 h-24 bg-light-muted-text/10 dark:bg-dark-muted-text/10 rounded-3xl flex items-center justify-center mx-auto">
                                                <FileVideo className="w-12 h-12 text-light-muted-text dark:text-dark-muted-text" />
                                            </div>
                                            <div>
                                                <p className="text-2xl font-semibold text-light-text dark:text-dark-text mb-2">
                                                    No video selected
                                                </p>
                                                <p className="text-lg text-light-muted-text dark:text-dark-muted-text">
                                                    Upload a video to see
                                                    preview and controls
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Enhanced Video Metadata */}
                            {videoFile && (
                                <div className="mt-8 p-6 bg-light-muted-background/30 dark:bg-dark-muted-background/30 rounded-2xl">
                                    <h3 className="text-lg font-bold text-light-text dark:text-dark-text mb-4 flex items-center">
                                        <Settings className="w-5 h-5 mr-2" />
                                        File Information
                                    </h3>
                                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
                                        {[
                                            {
                                                label: "File Name",
                                                value: videoFile.name,
                                                truncate: true,
                                            },
                                            {
                                                label: "File Size",
                                                value: formatFileSize(
                                                    videoFile.size
                                                ),
                                            },
                                            {
                                                label: "File Type",
                                                value: videoFile.type,
                                            },
                                            {
                                                label: "Last Modified",
                                                value: new Date(
                                                    videoFile.lastModified
                                                ).toLocaleDateString(),
                                            },
                                        ].map((item, index) => (
                                            <div key={index}>
                                                <span className="text-sm font-medium text-light-muted-text dark:text-dark-muted-text block mb-1">
                                                    {item.label}:
                                                </span>
                                                <p
                                                    className={`text-light-text dark:text-dark-text font-semibold ${
                                                        item.truncate
                                                            ? "truncate"
                                                            : ""
                                                    }`}
                                                    title={
                                                        item.truncate
                                                            ? item.value
                                                            : undefined
                                                    }
                                                >
                                                    {item.value}
                                                </p>
                                            </div>
                                        ))}
                                    </div>
                                    {duration > 0 && (
                                        <div className="mt-4 pt-4 border-t border-light-muted-text/10 dark:border-dark-muted-text/10">
                                            <div className="flex items-center justify-between">
                                                <span className="text-sm font-medium text-light-muted-text dark:text-dark-muted-text">
                                                    Duration:
                                                </span>
                                                <span className="text-light-text dark:text-dark-text font-semibold">
                                                    {formatTime(duration)}
                                                </span>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default UploadPage;
