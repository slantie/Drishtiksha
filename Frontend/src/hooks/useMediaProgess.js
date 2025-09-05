// src/hooks/useMediaProgess.js

import { useState, useEffect, useRef } from "react"; // Added useRef
import { socketService } from "../lib/socket.jsx"; // Ensure correct path and .jsx extension

/**
 * A React hook to track real-time media processing progress via WebSockets.
 * It listens for 'progress_update' events and filters them by mediaId.
 *
 * @param {string} mediaId - The ID of the media item to track.
 * @returns {{latestProgress: object|null, progressEvents: object[]}} - An object containing
 *   the latest progress event and an array of all received progress events for this mediaId.
 */
export const useMediaProgress = (mediaId) => {
  const [progressEvents, setProgressEvents] = useState([]);
  const isMounted = useRef(true); // To prevent state updates on unmounted component

  useEffect(() => {
    isMounted.current = true;
    setProgressEvents([]); // Clear events when mediaId changes or component mounts

    if (!socketService.socket || !mediaId) {
      console.log(`[useMediaProgress] Skipping setup: socket not ready or mediaId missing (mediaId: ${mediaId})`);
      return;
    }

    const handleProgress = (data) => {
      // The forwarded Python event has 'media_id'. We check if it matches.
      // Backend now sends 'mediaId' directly in event payload, as confirmed in Batch 3 (socket.jsx).
      if (isMounted.current && data.mediaId === mediaId) { // Use 'mediaId'
        // console.log(`[useMediaProgress] Received relevant progress for ${mediaId}:`, data);
        setProgressEvents((prevEvents) => [...prevEvents, data]);
      } else {
        // console.log(`[useMediaProgress] Received irrelevant progress (target: ${mediaId}, event mediaId: ${data.mediaId})`, data);
      }
    };

    // Ensure we only subscribe if the socket is connected and authenticated
    if (socketService.socket.connected) {
      socketService.socket.on("progress_update", handleProgress);
      // console.log(`[useMediaProgress] Subscribed to 'progress_update' for mediaId: ${mediaId}`);
    } else {
      // If socket is not connected, try to reconnect or log a warning.
      // The AuthContext already handles connecting/disconnecting the socket.
      console.warn(`[useMediaProgress] Socket not connected for mediaId: ${mediaId}. Progress updates may be delayed.`);
    }

    return () => {
      if (socketService.socket) {
        socketService.socket.off("progress_update", handleProgress);
        // console.log(`[useMediaProgress] Unsubscribed from 'progress_update' for mediaId: ${mediaId}`);
      }
      isMounted.current = false;
    };
  }, [mediaId]); // Dependency on mediaId ensures re-subscription when tracking a different media item

  const latestProgress =
    progressEvents.length > 0
      ? progressEvents[progressEvents.length - 1]
      : null;

  return { latestProgress, progressEvents };
};