// src/hooks/useMediaProgress.js

import { useState, useEffect } from "react";
import { socketService } from "../lib/socket";

// RENAMED: from useVideoProgress to useMediaProgress
export const useMediaProgress = (mediaId) => {
    const [progressEvents, setProgressEvents] = useState([]);

    useEffect(() => {
        // UPDATED: Check for mediaId
        if (!socketService.socket || !mediaId) return;

        const handleProgress = (data) => {
            // The backend socket event sends a 'videoId' property which is the mediaId.
            // We handle it here and check against our required mediaId.
            const eventMediaId = data.videoId || data.mediaId;
            if (eventMediaId === mediaId) {
                setProgressEvents((prevEvents) => [...prevEvents, data]);
            }
        };

        socketService.socket.on("progress_update", handleProgress);

        return () => {
            socketService.socket.off("progress_update", handleProgress);
        };
    // UPDATED: Dependency is now mediaId
    }, [mediaId]);

    const latestProgress =
        progressEvents.length > 0
            ? progressEvents[progressEvents.length - 1]
            : null;

    return { latestProgress, progressEvents };
};