// src/components/videos/VideoPlayer.jsx

import React, { useState, useEffect, useRef } from "react";
import {
    Play,
    Pause,
    Volume2,
    VolumeX,
    Maximize,
    Loader2,
    VolumeOff,
} from "lucide-react";

const getOptimizedVideoUrl = (url) => {
    if (!url) return null;
    if (url.includes("/upload/") && !url.includes("/upload/f_auto,q_auto/")) {
        return url.replace("/upload/", "/upload/f_auto,q_auto/");
    }
    return url;
};

const formatTime = (timeInSeconds) => {
    if (isNaN(timeInSeconds)) return "0:00";
    const minutes = Math.floor(timeInSeconds / 60);
    const seconds = Math.floor(timeInSeconds % 60);
    return `${minutes}:${seconds.toString().padStart(2, "0")}`;
};

export const VideoPlayer = ({ videoUrl }) => {
    const videoRef = useRef(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [volume, setVolume] = useState(1);
    const [isMuted, setIsMuted] = useState(false);
    const [isLoading, setIsLoading] = useState(true);
    const [hasAudio, setHasAudio] = useState(true);

    const togglePlayPause = () => {
        if (!videoRef.current) return;
        if (isPlaying) {
            videoRef.current.pause();
        } else {
            videoRef.current.play();
        }
    };

    const handleSeek = (e) => {
        if (!videoRef.current || !duration || duration === 0) return;

        const rect = e.currentTarget.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const percentage = Math.max(0, Math.min(1, clickX / rect.width));
        const newTime = percentage * duration;

        videoRef.current.currentTime = newTime;
        setCurrentTime(newTime);
    };

    const handleVolumeChange = (e) => {
        const newVolume = parseFloat(e.target.value);
        setVolume(newVolume);
        if (videoRef.current) videoRef.current.volume = newVolume;
        setIsMuted(newVolume === 0);
    };

    const toggleMute = () => {
        if (!videoRef.current) return;
        const newMuted = !isMuted;
        setIsMuted(newMuted);
        videoRef.current.muted = newMuted;
        if (!newMuted && volume === 0) {
            setVolume(1);
            videoRef.current.volume = 1;
        }
    };

    const toggleFullscreen = () => {
        if (!videoRef.current) return;
        if (!document.fullscreenElement) {
            videoRef.current.parentElement.requestFullscreen?.();
        } else {
            document.exitFullscreen?.();
        }
    };

    useEffect(() => {
        const video = videoRef.current;
        if (!video) return;

        const onTimeUpdate = () => setCurrentTime(video.currentTime);
        const onLoadedMetadata = () => {
            setDuration(video.duration);
            setIsLoading(false);

            // Check if video has audio tracks
            let hasAudioTracks = false;

            // Method 1: Check audioTracks
            if (video.audioTracks && video.audioTracks.length > 0) {
                hasAudioTracks = true;
            }

            // Method 2: Check webkitAudioDecodedByteCount (Safari/Chrome)
            if (
                video.webkitAudioDecodedByteCount !== undefined &&
                video.webkitAudioDecodedByteCount > 0
            ) {
                hasAudioTracks = true;
            }

            // Method 3: Check mozHasAudio (Firefox)
            if (video.mozHasAudio === true) {
                hasAudioTracks = true;
            }

            setHasAudio(hasAudioTracks);
        };

        const onLoadedData = () => {
            // Additional check once data is fully loaded
            // This is a simpler check that doesn't depend on state
            setTimeout(() => {
                // Allow a brief moment for audio detection
                if (
                    video.webkitAudioDecodedByteCount === 0 &&
                    (!video.audioTracks || video.audioTracks.length === 0) &&
                    video.mozHasAudio !== true
                ) {
                    setHasAudio(false);
                }
            }, 100);
        };
        const onPlay = () => setIsPlaying(true);
        const onPause = () => setIsPlaying(false);
        const onWaiting = () => setIsLoading(true);
        const onPlaying = () => setIsLoading(false);
        const onEnded = () => {
            setIsPlaying(false);
            setCurrentTime(0);
        };

        video.addEventListener("timeupdate", onTimeUpdate);
        video.addEventListener("loadedmetadata", onLoadedMetadata);
        video.addEventListener("loadeddata", onLoadedData);
        video.addEventListener("play", onPlay);
        video.addEventListener("pause", onPause);
        video.addEventListener("waiting", onWaiting);
        video.addEventListener("playing", onPlaying);
        video.addEventListener("ended", onEnded);

        return () => {
            video.removeEventListener("timeupdate", onTimeUpdate);
            video.removeEventListener("loadedmetadata", onLoadedMetadata);
            video.removeEventListener("loadeddata", onLoadedData);
            video.removeEventListener("play", onPlay);
            video.removeEventListener("pause", onPause);
            video.removeEventListener("waiting", onWaiting);
            video.removeEventListener("playing", onPlaying);
            video.removeEventListener("ended", onEnded);
        };
    }, []);

    return (
        <div className="relative group bg-black rounded-lg overflow-hidden aspect-video">
            <video
                ref={videoRef}
                src={getOptimizedVideoUrl(videoUrl)}
                className="w-full h-full object-contain"
                onClick={togglePlayPause}
                onLoadedData={() => setIsLoading(false)}
            />

            {isLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                    <Loader2 className="w-8 h-8 text-white animate-spin" />
                </div>
            )}

            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                <div
                    className="w-full h-2 bg-white/30 rounded-full cursor-pointer mb-2 relative group/progress"
                    onClick={handleSeek}
                >
                    <div
                        className="h-full bg-red-500 rounded-full relative"
                        style={{
                            width: `${
                                duration ? (currentTime / duration) * 100 : 0
                            }%`,
                        }}
                    >
                        <div className="absolute right-0 top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full transform -translate-x-1/2 opacity-0 group-hover/progress:opacity-100 transition-opacity shadow-lg"></div>
                    </div>
                </div>
                <div className="flex items-center justify-between text-white">
                    <div className="flex items-center gap-3">
                        <button onClick={togglePlayPause}>
                            {isPlaying ? (
                                <Pause size={20} />
                            ) : (
                                <Play size={20} />
                            )}
                        </button>
                        <div className="flex items-center gap-2">
                            {!hasAudio ? (
                                <div className="flex items-center gap-1">
                                    <VolumeOff
                                        size={20}
                                        className="text-gray-400"
                                    />
                                    <span className="text-xs text-gray-400">
                                        No Audio Present
                                    </span>
                                </div>
                            ) : (
                                <>
                                    <button onClick={toggleMute}>
                                        {isMuted || volume === 0 ? (
                                            <VolumeX size={20} />
                                        ) : (
                                            <Volume2 size={20} />
                                        )}
                                    </button>
                                    <input
                                        type="range"
                                        min="0"
                                        max="1"
                                        step="0.1"
                                        value={isMuted ? 0 : volume}
                                        onChange={handleVolumeChange}
                                        className="w-20 h-1 accent-red-500"
                                    />
                                </>
                            )}
                        </div>
                        <span className="text-xs font-mono">
                            {formatTime(currentTime)} / {formatTime(duration)}
                        </span>
                    </div>
                    <button onClick={toggleFullscreen}>
                        <Maximize size={20} />
                    </button>
                </div>
            </div>
        </div>
    );
};
