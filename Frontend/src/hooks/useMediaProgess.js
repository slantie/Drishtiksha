// src/hooks/useMediaProgress.js

import { useState, useEffect } from "react";
import { socketService } from "../lib/socket";

export const useMediaProgress = (mediaId) => {
  const [progressEvents, setProgressEvents] = useState([]);

  useEffect(() => {
    if (!socketService.socket || !mediaId) return;

    const handleProgress = (data) => {
      // The forwarded Python event has 'media_id'. We check if it matches.
      if (data.media_id === mediaId) {
        setProgressEvents((prevEvents) => [...prevEvents, data]);
      }
    };

    socketService.socket.on("progress_update", handleProgress);

    return () => {
      socketService.socket.off("progress_update", handleProgress);
    };
  }, [mediaId]);

  const latestProgress =
    progressEvents.length > 0
      ? progressEvents[progressEvents.length - 1]
      : null;

  return { latestProgress, progressEvents };
};
