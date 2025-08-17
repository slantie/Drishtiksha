// src/hooks/useVideoProgress.js

import { useState, useEffect } from "react";
import { socketService } from "../lib/socket";

export const useVideoProgress = (videoId) => {
    const [progressEvents, setProgressEvents] = useState([]);

    useEffect(() => {
        if (!socketService.socket || !videoId) return;

        const handleProgress = (data) => {
            if (data.videoId === videoId) {
                // Add new event to the list, keeping a history
                setProgressEvents((prevEvents) => [...prevEvents, data]);
            }
        };

        socketService.socket.on("progress_update", handleProgress);

        // Cleanup on component unmount
        return () => {
            socketService.socket.off("progress_update", handleProgress);
        };
    }, [videoId]);

    // Return the latest progress event for simple display
    const latestProgress =
        progressEvents.length > 0
            ? progressEvents[progressEvents.length - 1]
            : null;

    return { latestProgress, progressEvents };
};
