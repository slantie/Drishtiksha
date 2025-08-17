// src/hooks/useVideoMetadata.js

import { useState, useEffect } from "react";

export const useVideoMetadata = (videoUrl, frameCount) => {
    const [fps, setFps] = useState(30); // Default to 30fps

    useEffect(() => {
        if (!videoUrl || !frameCount) return;

        const videoElement = document.createElement("video");
        videoElement.src = videoUrl;
        videoElement.preload = "metadata";

        const handleMetadata = () => {
            if (videoElement.duration && !isNaN(videoElement.duration)) {
                const calculatedFps = frameCount / videoElement.duration;
                // Use a reasonable range to avoid extreme values from very short videos
                if (calculatedFps > 10 && calculatedFps < 120) {
                    setFps(Math.round(calculatedFps));
                }
            }
            // Clean up the element after use
            videoElement.remove();
        };

        const handleError = () => {
            console.warn(
                "Could not load video metadata to calculate FPS. Falling back to 30fps."
            );
            videoElement.remove();
        };

        videoElement.addEventListener("loadedmetadata", handleMetadata);
        videoElement.addEventListener("error", handleError);

        // Cleanup function
        return () => {
            videoElement.removeEventListener("loadedmetadata", handleMetadata);
            videoElement.removeEventListener("error", handleError);
            videoElement.remove();
        };
    }, [videoUrl, frameCount]);

    return fps;
};
