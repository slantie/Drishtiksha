// Frontend/src/components/media/MediaPlayer.jsx

import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  Play,
  Pause,
  Volume2,
  VolumeX,
  Maximize,
  Loader2,
  Music,
  AlertTriangle,
} from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "../ui/DropdownMenu";
import { Button } from "../ui/Button";

const formatTime = (timeInSeconds) => {
  if (isNaN(timeInSeconds) || timeInSeconds < 0) return "00:00";
  const minutes = Math.floor(timeInSeconds / 60);
  const seconds = Math.floor(timeInSeconds % 60);
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(
    2,
    "0"
  )}`;
};

export const MediaPlayer = ({ media }) => {
  console.log("[MediaPlayer] Rendering with media prop:", media);

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

  const isPlayable =
    media?.mediaType === "VIDEO" || media?.mediaType === "AUDIO";
  const playbackRates = [0.5, 0.75, 1, 1.25, 1.5, 2];

  const togglePlayPause = useCallback(() => {
    if (!mediaRef.current || !isPlayable) return;
    console.log(
      "[MediaPlayer] Toggling play/pause. Current state:",
      mediaRef.current.paused ? "paused" : "playing"
    );
    mediaRef.current.paused
      ? mediaRef.current.play()
      : mediaRef.current.pause();
  }, [isPlayable]);

  const toggleFullScreen = useCallback(() => {
    if (media?.mediaType !== "VIDEO") return;
    if (!document.fullscreenElement) {
      console.log("[MediaPlayer] Entering fullscreen.");
      containerRef.current?.requestFullscreen?.();
    } else {
      console.log("[MediaPlayer] Exiting fullscreen.");
      document.exitFullscreen?.();
    }
  }, [media?.mediaType]);

  const handleSeek = (e) => {
    if (!mediaRef.current || !isPlayable) return;
    const progressBar = e.currentTarget;
    const rect = progressBar.getBoundingClientRect();
    const seekPosition = (e.clientX - rect.left) / rect.width;
    const seekTime = seekPosition * duration;
    console.log(`[MediaPlayer] Seeking to ${seekTime.toFixed(2)}s`);
    mediaRef.current.currentTime = seekTime;
  };

  useEffect(() => {
    const element = mediaRef.current;
    if (!element) return;

    const onLoadedData = () => {
      console.log("[MediaPlayer] Event: loadeddata");
      if (isPlayable) setDuration(element.duration);
      setIsLoading(false);
    };
    const onTimeUpdate = () => setCurrentTime(element.currentTime);
    const onPlay = () => {
      console.log("[MediaPlayer] Event: play");
      setIsPlaying(true);
    };
    const onPause = () => {
      console.log("[MediaPlayer] Event: pause");
      setIsPlaying(false);
    };
    const onVolumeChange = () => {
      setVolume(element.volume);
      setIsMuted(element.muted);
    };
    const onError = () => {
      console.error("[MediaPlayer] Event: error - Failed to load media.");
      setIsLoading(false);
    };

    element.addEventListener("loadeddata", onLoadedData);
    element.addEventListener("timeupdate", onTimeUpdate);
    element.addEventListener("play", onPlay);
    element.addEventListener("pause", onPause);
    element.addEventListener("volumechange", onVolumeChange);
    element.addEventListener("error", onError);

    return () => {
      element.removeEventListener("loadeddata", onLoadedData);
      element.removeEventListener("timeupdate", onTimeUpdate);
      element.removeEventListener("play", onPlay);
      element.removeEventListener("pause", onPause);
      element.removeEventListener("volumechange", onVolumeChange);
      element.removeEventListener("error", onError);
    };
  }, [isPlayable]);

  useEffect(() => {
    const onFullScreenChange = () =>
      setIsFullScreen(!!document.fullscreenElement);
    document.addEventListener("fullscreenchange", onFullScreenChange);
    return () =>
      document.removeEventListener("fullscreenchange", onFullScreenChange);
  }, []);

  const renderMediaElement = () => {
    console.log(
      `[MediaPlayer] Rendering media element for type: ${media.mediaType}, URL: ${media.url}`
    );
    switch (media.mediaType) {
      case "VIDEO":
        return (
          <video
            ref={mediaRef}
            src={media.url}
            className="w-full h-full object-contain"
            onDoubleClick={toggleFullScreen}
          />
        );
      case "AUDIO":
        return <audio ref={mediaRef} src={media.url} />;
      case "IMAGE":
        return (
          <img
            src={media.url}
            alt={media.description || media.filename}
            className="w-full h-full object-contain"
          />
        );
      default:
        return <div className="text-white">Unsupported media type</div>;
    }
  };

  return (
    <div
      ref={containerRef}
      className="relative group bg-black rounded-lg overflow-hidden aspect-video focus:outline-none"
      tabIndex={0}
    >
      {media.mediaType === "AUDIO" ? (
        <div className="w-full h-full flex flex-col items-center justify-center bg-black p-4">
          <Music className="w-24 h-24 text-gray-500 mb-4" />
          <p className="text-white text-center font-semibold">
            {media.filename}
          </p>
        </div>
      ) : (
        renderMediaElement()
      )}

      {isLoading && isPlayable && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50">
          <Loader2 className="w-8 h-8 text-white animate-spin" />
        </div>
      )}

      {isPlayable && (
        <div
          className={`absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-3 transition-opacity duration-300 ${
            media.mediaType === "AUDIO"
              ? "opacity-100"
              : "opacity-0 group-hover:opacity-100 group-focus-within:opacity-100"
          }`}
        >
          <div
            className="relative h-2 w-full cursor-pointer group/progress"
            onClick={handleSeek}
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
              <Button variant="ghost" size="icon" onClick={togglePlayPause}>
                {isPlaying ? <Pause /> : <Play />}
              </Button>
              <div className="flex items-center gap-2 group/volume">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => (mediaRef.current.muted = !isMuted)}
                >
                  {isMuted || volume === 0 ? <VolumeX /> : <Volume2 />}
                </Button>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={isMuted ? 0 : volume}
                  onChange={(e) =>
                    (mediaRef.current.volume = parseFloat(e.target.value))
                  }
                  className="w-0 group-hover/volume:w-20 h-1 accent-primary-main transition-all duration-300"
                />
              </div>
              <span className="text-sm font-mono">
                {formatTime(currentTime)} / {formatTime(duration)}
              </span>
            </div>
            <div className="flex items-center gap-1">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" className="w-20 font-mono">
                    {playbackRate}x
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent>
                  {playbackRates.map((rate) => (
                    <DropdownMenuItem
                      key={rate}
                      onSelect={() => {
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
                <Button variant="ghost" size="icon" onClick={toggleFullScreen}>
                  <Maximize />
                </Button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
