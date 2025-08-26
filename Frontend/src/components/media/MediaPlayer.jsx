// src/components/media/MediaPlayer.jsx

import React, { useState, useEffect, useRef } from "react";
import { Play, Pause, Volume2, VolumeX, Maximize, Loader2, Music, Image as ImageIcon, AlertTriangle } from "lucide-react";
import { cn } from "../../lib/utils";

// Helper function to get optimized Cloudinary URLs
const getOptimizedUrl = (url, mediaType) => {
    if (!url || !url.includes("/upload/")) return url;
    
    // Add standard f_auto,q_auto for all types
    if (!url.includes("/upload/f_auto,q_auto")) {
        url = url.replace("/upload/", "/upload/f_auto,q_auto/");
    }
    
    // For video, we can add streaming profile transformations
    if (mediaType === 'VIDEO' && !url.includes('sp_auto')) {
       url = url.replace("/upload/", "/upload/sp_auto/");
    }
    return url;
};

const formatTime = (timeInSeconds) => {
    if (isNaN(timeInSeconds)) return "00:00";
    const minutes = Math.floor(timeInSeconds / 60);
    const seconds = Math.floor(timeInSeconds % 60);
    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
};

// RENAMED: from VideoPlayer to MediaPlayer
export const MediaPlayer = ({ media }) => {
    if (!media) {
        return (
            <div className="relative group bg-black rounded-lg overflow-hidden aspect-video flex items-center justify-center text-red-500">
                <AlertTriangle className="w-8 h-8 mr-2" />
                <p>Media data not available.</p>
            </div>
        );
    }

    const mediaRef = useRef(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [volume, setVolume] = useState(1);
    const [isLoading, setIsLoading] = useState(true);

    const isPlayable = media.mediaType === 'VIDEO' || media.mediaType === 'AUDIO';

    const togglePlayPause = () => {
        if (!mediaRef.current || !isPlayable) return;
        mediaRef.current.paused ? mediaRef.current.play() : mediaRef.current.pause();
    };

    const handleSeek = (e) => {
        if (!mediaRef.current || !isPlayable) return;
        const seekTime = (e.nativeEvent.offsetX / e.currentTarget.offsetWidth) * duration;
        mediaRef.current.currentTime = seekTime;
    };

    const handleVolumeChange = (e) => {
        if (!mediaRef.current || !isPlayable) return;
        const newVolume = parseFloat(e.target.value);
        mediaRef.current.volume = newVolume;
        setVolume(newVolume);
    };

    // Fullscreen is only for video
    const toggleFullscreen = () => {
        if (media.mediaType !== 'VIDEO' || !mediaRef.current) return;
        // Fullscreen logic remains the same...
        if (mediaRef.current.requestFullscreen) {
            mediaRef.current.requestFullscreen();
        }
    };

    useEffect(() => {
        const element = mediaRef.current;
        if (!element) return;
        
        // Generic event handlers
        const onLoadedData = () => {
            if (isPlayable) setDuration(element.duration);
            setIsLoading(false);
        };
        const onWaiting = () => setIsLoading(true);
        const onPlaying = () => setIsLoading(false);
        const onPlay = () => setIsPlaying(true);
        const onPause = () => setIsPlaying(false);
        const onTimeUpdate = () => setCurrentTime(element.currentTime);

        element.addEventListener("loadeddata", onLoadedData);
        element.addEventListener("waiting", onWaiting);
        element.addEventListener("playing", onPlaying);
        if (isPlayable) {
            element.addEventListener("play", onPlay);
            element.addEventListener("pause", onPause);
            element.addEventListener("timeupdate", onTimeUpdate);
        }

        return () => {
            element.removeEventListener("loadeddata", onLoadedData);
            element.removeEventListener("waiting", onWaiting);
            element.removeEventListener("playing", onPlaying);
            if (isPlayable) {
                element.removeEventListener("play", onPlay);
                element.removeEventListener("pause", onPause);
                element.removeEventListener("timeupdate", onTimeUpdate);
            }
        };
    }, [isPlayable]);

    // CONDITIONAL RENDERING LOGIC
    const renderMediaElement = () => {
        const optimizedUrl = getOptimizedUrl(media.url, media.mediaType);
        switch (media.mediaType) {
            case 'VIDEO':
                return <video ref={mediaRef} src={optimizedUrl} className="w-full h-full object-contain" onClick={togglePlayPause} />;
            case 'AUDIO':
                return (
                    <div className="w-full h-full flex flex-col items-center justify-center bg-black p-4">
                        <Music className="w-24 h-24 text-gray-500 mb-4" />
                        <audio ref={mediaRef} src={optimizedUrl} className="hidden" />
                        <p className="text-white text-center font-semibold">{media.filename}</p>
                    </div>
                );
            case 'IMAGE':
                return <img ref={mediaRef} src={optimizedUrl} alt={media.description || media.filename} className="w-full h-full object-contain" />;
            default:
                return <div className="text-white">Unsupported media type</div>;
        }
    };

    return (
        <div className="relative group bg-black rounded-lg overflow-hidden aspect-video">
            {renderMediaElement()}
            
            {isLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                    <Loader2 className="w-8 h-8 text-white animate-spin" />
                </div>
            )}
            
            {isPlayable && (
                <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-4 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                    <div className="relative h-1.5 w-full cursor-pointer" onClick={handleSeek}>
                        <div className="absolute w-full h-full bg-white/30 rounded-full top-0"></div>
                        <div className="absolute h-full bg-primary-main rounded-full" style={{ width: `${(currentTime / duration) * 100}%` }}></div>
                    </div>
                    <div className="flex items-center justify-between text-white mt-2">
                        <div className="flex items-center gap-4">
                            <button onClick={togglePlayPause}>{isPlaying ? <Pause size={20} /> : <Play size={20} />}</button>
                            <div className="flex items-center gap-2">
                                <button onClick={() => handleVolumeChange({ target: { value: volume > 0 ? 0 : 1 } })}>
                                    {volume > 0 ? <Volume2 size={20} /> : <VolumeX size={20} />}
                                </button>
                                <input type="range" min="0" max="1" step="0.05" value={volume} onChange={handleVolumeChange} className="w-20 h-1 accent-primary-main" />
                            </div>
                            <span className="text-xs font-mono">{formatTime(currentTime)} / {formatTime(duration)}</span>
                        </div>
                        {media.mediaType === 'VIDEO' && (
                            <button onClick={toggleFullscreen}><Maximize size={20} /></button>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};