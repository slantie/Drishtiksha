// Frontend/src/components/media/MediaPlayer.jsx

import React, { useReducer, useEffect, useRef, useCallback } from "react";
import {
  Play,
  Pause,
  Volume2,
  VolumeX,
  Maximize,
  Minimize,
  Loader2,
  Music,
  XCircle,
  Image as ImageIcon,
} from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "../ui/DropdownMenu"; // Make sure path is correct
import { Button } from "../ui/Button"; // Make sure path is correct
import { Slider } from "../ui/Slider"; // <-- Importing our custom slider
import showToast from "../../utils/toast";

const formatTime = (timeInSeconds) => {
  if (isNaN(timeInSeconds) || !isFinite(timeInSeconds)) return "00:00";
  const minutes = Math.floor(timeInSeconds / 60);
  const seconds = Math.floor(timeInSeconds % 60);
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(
    2,
    "0"
  )}`;
};

// Reducer for centralizing player state management
const initialState = {
  isPlaying: false,
  currentTime: 0,
  duration: 0,
  volume: 1,
  isMuted: false,
  playbackRate: 1,
  isLoading: true,
  isFullScreen: false,
  hasError: false,
};

function playerReducer(state, action) {
  switch (action.type) {
    case "RESET":
      return { ...initialState, volume: state.volume, isMuted: state.isMuted };
    case "PLAY":
      return { ...state, isPlaying: true };
    case "PAUSE":
      return { ...state, isPlaying: false };
    case "TIME_UPDATE":
      return { ...state, currentTime: action.payload };
    case "LOADED_METADATA":
      return { ...state, duration: action.payload, isLoading: false };
    case "VOLUME_CHANGE":
      return {
        ...state,
        volume: action.payload.volume,
        isMuted: action.payload.isMuted,
      };
    case "SET_PLAYBACK_RATE":
      return { ...state, playbackRate: action.payload };
    case "SET_LOADING":
      return { ...state, isLoading: action.payload };
    case "SET_FULLSCREEN":
      return { ...state, isFullScreen: action.payload };
    case "ERROR":
      return { ...state, hasError: true, isLoading: false };
    default:
      throw new Error(`Unhandled action type: ${action.type}`);
  }
}

export const MediaPlayer = ({ media }) => {
  const [state, dispatch] = useReducer(playerReducer, initialState);
  const {
    isPlaying,
    currentTime,
    duration,
    volume,
    isMuted,
    playbackRate,
    isLoading,
    isFullScreen,
    hasError,
  } = state;

  const mediaRef = useRef(null);
  const containerRef = useRef(null);

  const isPlayable =
    media?.mediaType === "VIDEO" || media?.mediaType === "AUDIO";
  const playbackRates = [0.5, 0.75, 1, 1.25, 1.5, 2];

  // Media Actions
  const togglePlayPause = useCallback(() => {
    if (!mediaRef.current || !isPlayable || hasError) return;
    if (mediaRef.current.paused) {
      mediaRef.current.play().catch(() => dispatch({ type: "ERROR" }));
    } else {
      mediaRef.current.pause();
    }
  }, [isPlayable, hasError]);

  const toggleFullScreen = useCallback(() => {
    if (media?.mediaType !== "VIDEO" || !containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().catch((err) => {
        console.error(
          `Error attempting to enable full-screen mode: ${err.message} (${err.name})`
        );
      });
    } else {
      document.exitFullscreen();
    }
  }, [media?.mediaType]);

  const handleSeek = (value) => {
    if (!mediaRef.current || !isPlayable || hasError || !isFinite(duration))
      return;
    const newTime = (value[0] / 100) * duration;
    mediaRef.current.currentTime = newTime;
    dispatch({ type: "TIME_UPDATE", payload: newTime });
  };

  const handleVolumeChange = (value) => {
    if (!mediaRef.current) return;
    const newVolume = value[0];
    mediaRef.current.muted = newVolume === 0;
    mediaRef.current.volume = newVolume;
  };

  // Keyboard controls effect
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")
        return;
      if (e.key === " ") {
        e.preventDefault();
        togglePlayPause();
      }
      if (e.key === "f" && media?.mediaType === "VIDEO") {
        e.preventDefault();
        toggleFullScreen();
      }
      if (e.key === "m" && isPlayable) {
        e.preventDefault();
        if (mediaRef.current) mediaRef.current.muted = !mediaRef.current.muted;
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [togglePlayPause, toggleFullScreen, isPlayable, media?.mediaType]);

  // Media element event listeners effect
  useEffect(() => {
    const element = mediaRef.current;
    if (!element || !isPlayable) return;

    dispatch({ type: "RESET" });

    const onLoadedData = () =>
      dispatch({ type: "LOADED_METADATA", payload: element.duration });
    const onTimeUpdate = () =>
      dispatch({ type: "TIME_UPDATE", payload: element.currentTime });
    const onPlay = () => dispatch({ type: "PLAY" });
    const onPause = () => dispatch({ type: "PAUSE" });
    const onVolumeChange = () =>
      dispatch({
        type: "VOLUME_CHANGE",
        payload: { volume: element.volume, isMuted: element.muted },
      });
    const onWaiting = () => dispatch({ type: "SET_LOADING", payload: true });
    const onPlaying = () => dispatch({ type: "SET_LOADING", payload: false });
    const onError = (e) => {
      console.error("[MediaPlayer] Failed to load media:", e);
      showToast.error(
        "Failed to load media. File may be corrupted or unsupported."
      );
      dispatch({ type: "ERROR" });
    };

    element.addEventListener("loadeddata", onLoadedData);
    element.addEventListener("timeupdate", onTimeUpdate);
    element.addEventListener("play", onPlay);
    element.addEventListener("pause", onPause);
    element.addEventListener("volumechange", onVolumeChange);
    element.addEventListener("waiting", onWaiting);
    element.addEventListener("playing", onPlaying);
    element.addEventListener("ended", onPause);
    element.addEventListener("error", onError);

    return () => {
      element.removeEventListener("loadeddata", onLoadedData);
      element.removeEventListener("timeupdate", onTimeUpdate);
      element.removeEventListener("play", onPlay);
      element.removeEventListener("pause", onPause);
      element.removeEventListener("volumechange", onVolumeChange);
      element.removeEventListener("waiting", onWaiting);
      element.removeEventListener("playing", onPlaying);
      element.removeEventListener("ended", onPause);
      element.removeEventListener("error", onError);
    };
  }, [isPlayable, media.url]);

  // Fullscreen change listener effect
  useEffect(() => {
    const onFullScreenChange = () =>
      dispatch({
        type: "SET_FULLSCREEN",
        payload: !!document.fullscreenElement,
      });
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
            key={media.url}
            ref={mediaRef}
            src={media.url}
            className="w-full h-full object-contain bg-black"
            onDoubleClick={toggleFullScreen}
            preload="metadata"
          />
        );
      case "AUDIO":
        return (
          <audio
            key={media.url}
            ref={mediaRef}
            src={media.url}
            preload="metadata"
          />
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
          ? "aspect-video max-h-[400px]"
          : media.mediaType === "AUDIO"
          ? "aspect-video max-h-[280px]"
          : "aspect-square max-h-[400px]"
      }`}
      tabIndex={0}
    >
      {media.mediaType === "AUDIO" ? (
        <div className="w-full h-full flex flex-col items-center justify-center bg-black p-4">
          <Music className="w-24 h-24 text-gray-500 mb-4" />
          <p className="text-white text-center font-semibold">
            {media.filename || "Audio File"}
          </p>
        </div>
      ) : (
        renderMediaElement()
      )}

      {isLoading && isPlayable && !hasError && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50 pointer-events-none">
          <Loader2 className="w-8 h-8 text-white animate-spin" />
        </div>
      )}

      {hasError && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/70 text-red-400">
          <XCircle className="w-10 h-10 mb-2" />
          <p>Error Loading Media</p>
        </div>
      )}

      {isPlayable && !hasError && (
        <div
          className={`absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-3 transition-opacity duration-300 ${
            media.mediaType === "AUDIO" || isPlaying === false
              ? "opacity-100"
              : "opacity-0 group-hover:opacity-100 group-focus-within:opacity-100"
          }`}
        >
          <Slider
            value={[(currentTime / duration) * 100 || 0]}
            onValueChange={handleSeek}
            max={100}
            step={0.1}
            className="w-full h-2"
            aria-label="Seek media"
          />

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
                <Slider
                  value={[isMuted ? 0 : volume]}
                  onValueChange={handleVolumeChange}
                  max={1}
                  step={0.05}
                  className="w-0 group-hover/volume:w-20 h-1 transition-all duration-300"
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
                    className="w-20"
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
                        dispatch({ type: "SET_PLAYBACK_RATE", payload: rate });
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
