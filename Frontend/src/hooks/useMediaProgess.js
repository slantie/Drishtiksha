// src/hooks/useMediaProgress.js

import { useState, useEffect } from "react";
import { socketService } from "../lib/socket";

// RENAMED: from useVideoProgress to useMediaProgress
// PARAMETER RENAMED: from videoId to mediaId for clarity
export const useMediaProgress = (mediaId) => {
    const [progressEvents, setProgressEvents] = useState([]);

    useEffect(() => {
        // UPDATED: Check for mediaId
        if (!socketService.socket || !mediaId) return;

        const handleProgress = (data) => {
            // The backend socket event sends a 'videoId' property in its payload.
            // We treat this as our generic mediaId and check if it matches the one
            // this hook instance is listening for.
            const eventMediaId = data.videoId; 
            if (eventMediaId === mediaId) {
                // Add new event to the list, keeping a history
                setProgressEvents((prevEvents) => [...prevEvents, data]);
            }
        };

        socketService.socket.on("progress_update", handleProgress);

        // Cleanup on component unmount
        return () => {
            socketService.socket.off("progress_update", handleProgress);
        };
    // UPDATED: The hook's dependency is now mediaId
    }, [mediaId]);

    // Return the latest progress event for simple display
    const latestProgress =
        progressEvents.length > 0
            ? progressEvents[progressEvents.length - 1]
            : null;

    return { latestProgress, progressEvents };
};