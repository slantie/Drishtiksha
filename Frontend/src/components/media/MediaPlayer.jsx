// Frontend/src/components/media/MediaPlayer.jsx

import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  Play,
  Pause,
  Volume2,
  VolumeX,
  XCircle,
  Maximize,
  Minimize, // Added Minimize icon for fullscreen toggle
  Loader2,
  Music,
  Video, // Added Video icon for video media type
  Image as ImageIcon, // Added ImageIcon for image media type
} from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "../ui/DropdownMenu";
import { Button } from "../ui/Button";
import showToast from "../../utils/toast";

const formatTime = (timeInSeconds) => {
  if (isNaN(timeInSeconds) || timeInSeconds < 0 || !isFinite(timeInSeconds))
    return "00:00"; // Handle Infinity/NaN
  const minutes = Math.floor(timeInSeconds / 60);
  const seconds = Math.floor(timeInSeconds % 60);
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(
    2,
    "0"
  )}`;
};

export const MediaPlayer = ({ media }) => {
  const mediaRef = useRef(null);
  const containerRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [isLoading, setIsLoading] = useState(true);
  const [isFullScreen, setIsFullScreen] = useState(false);
  const [hasError, setHasError] = useState(false); // New state for media loading errors

  const isPlayable =
    media?.mediaType === "VIDEO" || media?.mediaType === "AUDIO";
  const playbackRates = [0.5, 0.75, 1, 1.25, 1.5, 2];

  const togglePlayPause = useCallback(() => {
    if (!mediaRef.current || !isPlayable || hasError) return;
    mediaRef.current.paused
      ? mediaRef.current.play().catch((e) => {
          console.error("Error playing media:", e);
          setHasError(true);
          showToast.error("Failed to play media. Format may be unsupported.");
        })
      : mediaRef.current.pause();
  }, [isPlayable, hasError]);

  const toggleFullScreen = useCallback(() => {
    if (media?.mediaType !== "VIDEO" || !containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen();
    } else {
      document.exitFullscreen();
    }
  }, [media?.mediaType]);

  const handleSeek = (e) => {
    if (!mediaRef.current || !isPlayable || hasError) return;
    const progressBar = e.currentTarget;
    const rect = progressBar.getBoundingClientRect();
    const seekPosition = (e.clientX - rect.left) / rect.width;
    mediaRef.current.currentTime = seekPosition * duration;
  };

  // Keyboard controls
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Don't interfere with input fields
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")
        return;
      switch (e.key) {
        case " ":
          e.preventDefault();
          togglePlayPause();
          break;
        case "f":
          if (media?.mediaType === "VIDEO") {
            // Only for video
            e.preventDefault();
            toggleFullScreen();
          }
          break;
        case "m":
          e.preventDefault();
          if (isPlayable && mediaRef.current) {
            mediaRef.current.muted = !mediaRef.current.muted;
          }
          break;
        case "ArrowLeft": // Seek back 5 seconds
          e.preventDefault();
          if (isPlayable && mediaRef.current) {
            mediaRef.current.currentTime = Math.max(
              0,
              mediaRef.current.currentTime - 5
            );
          }
          break;
        case "ArrowRight": // Seek forward 5 seconds
          e.preventDefault();
          if (isPlayable && mediaRef.current) {
            mediaRef.current.currentTime = Math.min(
              duration,
              mediaRef.current.currentTime + 5
            );
          }
          break;
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
    togglePlayPause,
    toggleFullScreen,
    isPlayable,
    duration,
    media?.mediaType,
  ]);

  // Media element event listeners
  useEffect(() => {
    const element = mediaRef.current;
    if (!element || !isPlayable) return;

    // Reset states when media URL changes (via key prop) or component re-mounts
    setIsLoading(true);
    setHasError(false);
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(0);

    const onLoadedData = () => {
      setDuration(element.duration);
      setIsLoading(false);
    };
    const onTimeUpdate = () => setCurrentTime(element.currentTime);
    const onPlay = () => setIsPlaying(true);
    const onPause = () => setIsPlaying(false);
    const onVolumeChange = () => {
      setVolume(element.volume);
      setIsMuted(element.muted);
    };
    const onWaiting = () => setIsLoading(true); // Show loader when buffering
    const onPlaying = () => setIsLoading(false); // Hide loader when playing resumes
    const onEnded = () => setIsPlaying(false);
    const onError = (e) => {
      console.error("[MediaPlayer] Failed to load media:", e);
      setHasError(true);
      setIsLoading(false);
      showToast.error(
        "Failed to load media. The file may be corrupted or unsupported."
      );
    };

    element.addEventListener("loadeddata", onLoadedData);
    element.addEventListener("timeupdate", onTimeUpdate);
    element.addEventListener("play", onPlay);
    element.addEventListener("pause", onPause);
    element.addEventListener("volumechange", onVolumeChange);
    element.addEventListener("waiting", onWaiting);
    element.addEventListener("playing", onPlaying);
    element.addEventListener("ended", onEnded);
    element.addEventListener("error", onError);

    return () => {
      element.removeEventListener("loadeddata", onLoadedData);
      element.removeEventListener("timeupdate", onTimeUpdate);
      element.removeEventListener("play", onPlay);
      element.removeEventListener("pause", onPause);
      element.removeEventListener("volumechange", onVolumeChange);
      element.removeEventListener("waiting", onWaiting);
      element.removeEventListener("playing", onPlaying);
      element.removeEventListener("ended", onEnded);
      element.removeEventListener("error", onError);
    };
  }, [isPlayable, media.url]); // Re-run effect if media.url changes

  // Fullscreen change listener
  useEffect(() => {
    const onFullScreenChange = () =>
      setIsFullScreen(!!document.fullscreenElement);
    document.addEventListener("fullscreenchange", onFullScreenChange);
    return () =>
      document.removeEventListener("fullscreenchange", onFullScreenChange);
  }, []);

  const renderMediaElement = () => {
    if (!media?.url) {
      return (
        <div className="w-full h-full flex items-center justify-center bg-gray-900 text-gray-400">
          <ImageIcon className="w-16 h-16" />
          <p className="ml-4 text-xl">Media URL not available</p>
        </div>
      );
    }

    switch (media.mediaType) {
      case "VIDEO":
        return (
          <video
            key={media.url} // Key ensures remount on URL change
            ref={mediaRef}
            src={media.url}
            className="w-full h-full object-contain bg-black"
            onDoubleClick={toggleFullScreen}
            preload="metadata" // Load metadata to get duration quickly
          >
            Your browser does not support the video tag.
          </video>
        );
      case "AUDIO":
        return (
          // Audio element is hidden but its events drive custom controls
          <audio
            key={media.url}
            ref={mediaRef}
            src={media.url}
            preload="metadata"
          >
            Your browser does not support the audio tag.
          </audio>
        );
      case "IMAGE":
        return (
          <img
            key={media.url}
            src={media.url}
            alt={media.description || media.filename || "Uploaded image"}
            className="w-full h-full object-contain bg-black"
          />
        );
      default:
        return (
          <div className="w-full h-full flex items-center justify-center bg-gray-900 text-gray-400">
            <ImageIcon className="w-16 h-16" />
            <p className="ml-4 text-xl">Unsupported media type</p>
          </div>
        );
    }
  };

  return (
    <div
      ref={containerRef}
      className={`relative group bg-black rounded-lg overflow-hidden focus:outline-none ${
        media.mediaType === "VIDEO"
          ? "aspect-video"
          : "aspect-square max-h-[500px]"
      }`}
      tabIndex={0} // Makes the div focusable for keyboard events
    >
      {/* Media Display Area */}
      {media.mediaType === "AUDIO" ? (
        <div className="w-full h-full flex flex-col items-center justify-center bg-black p-4">
          <Music className="w-24 h-24 text-gray-500 mb-4" />
          <p className="text-white text-center font-semibold">
            {media.filename || "Audio File"}
          </p>
          {hasError && (
            <p className="text-red-500 text-sm mt-2">Error loading audio.</p>
          )}
        </div>
      ) : (
        renderMediaElement()
      )}

      {/* Loading Spinner / Error Overlay */}
      {isLoading && isPlayable && !hasError && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50">
          <Loader2 className="w-8 h-8 text-white animate-spin" />
        </div>
      )}
      {hasError && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/70 text-red-400">
          <XCircle className="w-10 h-10 mb-2" />
          <p>Error Loading Media</p>
          <p className="text-sm text-gray-400">Check file or URL.</p>
        </div>
      )}

      {/* Custom Controls (only for playable media, and if no error) */}
      {isPlayable && !hasError && (
        <div
          className={`absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-3 transition-opacity duration-300 ${
            media.mediaType === "AUDIO"
              ? "opacity-100" // Always visible for audio
              : "opacity-0 group-hover:opacity-100 group-focus-within:opacity-100"
          }`}
        >
          {/* Progress Bar */}
          <div
            className="relative h-2 w-full cursor-pointer group/progress"
            onClick={handleSeek}
            aria-label="Seek media"
          >
            <div className="absolute w-full h-1 bg-white/30 rounded-full top-1/2 -translate-y-1/2"></div>
            <div
              className="absolute h-1 bg-primary-main rounded-full top-1/2 -translate-y-1/2"
              style={{ width: `${(currentTime / duration) * 100}%` }}
            ></div>
            <div
              className="absolute h-3 w-3 bg-white rounded-full top-1/2 -translate-y-1/2 -translate-x-1/2 opacity-0 group-hover/progress:opacity-100 transition-opacity"
              style={{ left: `${(currentTime / duration) * 100}%` }}
            ></div>
          </div>

          <div className="flex items-center justify-between text-white mt-2">
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="icon"
                onClick={togglePlayPause}
                aria-label={isPlaying ? "Pause" : "Play"}
              >
                {isPlaying ? (
                  <Pause className="h-5 w-5" />
                ) : (
                  <Play className="h-5 w-5" />
                )}
              </Button>
              <div className="flex items-center gap-2 group/volume">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => {
                    if (mediaRef.current) mediaRef.current.muted = !isMuted;
                  }}
                  aria-label={isMuted ? "Unmute" : "Mute"}
                >
                  {isMuted || volume === 0 ? (
                    <VolumeX className="h-5 w-5" />
                  ) : (
                    <Volume2 className="h-5 w-5" />
                  )}
                </Button>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={isMuted ? 0 : volume}
                  onChange={(e) => {
                    if (mediaRef.current)
                      mediaRef.current.volume = parseFloat(e.target.value);
                  }}
                  className="w-0 group-hover/volume:w-20 h-1 accent-primary-main transition-all duration-300"
                  aria-label="Volume control"
                />
              </div>
              <span className="text-sm font-mono">
                {formatTime(currentTime)} / {formatTime(duration)}
              </span>
            </div>
            <div className="flex items-center gap-1">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="ghost"
                    className="w-20 font-mono"
                    aria-label="Playback speed"
                  >
                    {playbackRate}x
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent>
                  {playbackRates.map((rate) => (
                    <DropdownMenuItem
                      key={rate}
                      onSelect={() => {
                        if (mediaRef.current)
                          mediaRef.current.playbackRate = rate;
                        setPlaybackRate(rate);
                      }}
                    >
                      {rate}x
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
              {media.mediaType === "VIDEO" && (
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={toggleFullScreen}
                  aria-label={
                    isFullScreen ? "Exit fullscreen" : "Enter fullscreen"
                  }
                >
                  {isFullScreen ? (
                    <Minimize className="h-5 w-5" />
                  ) : (
                    <Maximize className="h-5 w-5" />
                  )}
                </Button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
