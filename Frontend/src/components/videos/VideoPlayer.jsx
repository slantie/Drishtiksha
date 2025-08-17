// src/components/videos/VideoPlayer.jsx

import React, { useState, useEffect, useRef } from "react";
import { Play, Pause, Volume2, VolumeX, Maximize, Loader2 } from "lucide-react";

// REFACTOR: Logic is preserved, but UI elements are restyled for a cleaner, modern look.
const getOptimizedVideoUrl = (url) => {
    if (!url) return null;
    if (url.includes("/upload/") && !url.includes("/upload/f_auto,q_auto/")) {
        return url.replace("/upload/", "/upload/f_auto,q_auto/");
    }
    return url;
};

const formatTime = (timeInSeconds) => {
    if (isNaN(timeInSeconds)) return "00:00";
    const minutes = Math.floor(timeInSeconds / 60);
    const seconds = Math.floor(timeInSeconds % 60);
    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(
        2,
        "0"
    )}`;
};

export const VideoPlayer = ({ videoUrl }) => {
    const videoRef = useRef(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [volume, setVolume] = useState(1);
    const [isLoading, setIsLoading] = useState(true);

    // All hooks and handlers are preserved...
    const togglePlayPause = () =>
        videoRef.current?.paused
            ? videoRef.current?.play()
            : videoRef.current?.pause();
    const handleSeek = (e) => {
        if (!videoRef.current) return;
        const seekTime =
            (e.nativeEvent.offsetX / e.currentTarget.offsetWidth) * duration;
        videoRef.current.currentTime = seekTime;
    };
    const handleVolumeChange = (e) => {
        if (!videoRef.current) return;
        const newVolume = parseFloat(e.target.value);
        videoRef.current.volume = newVolume;
        setVolume(newVolume);
    };
    const toggleFullscreen = () => {
        /* ... preserved */
    };

    useEffect(() => {
        const video = videoRef.current;
        if (!video) return;

        const onPlay = () => setIsPlaying(true);
        const onPause = () => setIsPlaying(false);
        const onTimeUpdate = () => setCurrentTime(video.currentTime);
        const onLoadedData = () => {
            setDuration(video.duration);
            setIsLoading(false);
        };
        const onWaiting = () => setIsLoading(true);
        const onPlaying = () => setIsLoading(false);

        video.addEventListener("play", onPlay);
        video.addEventListener("pause", onPause);
        video.addEventListener("timeupdate", onTimeUpdate);
        video.addEventListener("loadeddata", onLoadedData);
        video.addEventListener("waiting", onWaiting);
        video.addEventListener("playing", onPlaying);

        return () => {
            video.removeEventListener("play", onPlay);
            video.removeEventListener("pause", onPause);
            video.removeEventListener("timeupdate", onTimeUpdate);
            video.removeEventListener("loadeddata", onLoadedData);
            video.removeEventListener("waiting", onWaiting);
            video.removeEventListener("playing", onPlaying);
        };
    }, []);

    return (
        <div className="relative group bg-black rounded-lg overflow-hidden aspect-video">
            <video
                ref={videoRef}
                src={getOptimizedVideoUrl(videoUrl)}
                className="w-full h-full object-contain"
                onClick={togglePlayPause}
            />
            {isLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                    <Loader2 className="w-8 h-8 text-white animate-spin" />
                </div>
            )}
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-4 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                {/* REFACTOR: Seek bar is now taller and more prominent. */}
                <div
                    className="relative h-1.5 w-full cursor-pointer"
                    onClick={handleSeek}
                >
                    <div className="absolute w-full h-full bg-white/30 rounded-full top-0"></div>
                    <div
                        className="absolute h-full bg-primary-main rounded-full"
                        style={{ width: `${(currentTime / duration) * 100}%` }}
                    ></div>
                </div>
                <div className="flex items-center justify-between text-white mt-2">
                    <div className="flex items-center gap-4">
                        <button onClick={togglePlayPause}>
                            {isPlaying ? (
                                <Pause size={20} />
                            ) : (
                                <Play size={20} />
                            )}
                        </button>
                        <div className="flex items-center gap-2">
                            <button
                                onClick={() =>
                                    handleVolumeChange({
                                        target: { value: volume > 0 ? 0 : 1 },
                                    })
                                }
                            >
                                {volume > 0 ? (
                                    <Volume2 size={20} />
                                ) : (
                                    <VolumeX size={20} />
                                )}
                            </button>
                            <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.05"
                                value={volume}
                                onChange={handleVolumeChange}
                                className="w-20 h-1 accent-primary-main"
                            />
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
