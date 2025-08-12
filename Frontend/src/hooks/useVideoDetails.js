// src/hooks/useVideoDetails.js

import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { videoApiService } from "../services/videoApiService.js";
import { showToast } from "../utils/toast.js";

export const useVideoDetails = (videoId) => {
    const [video, setVideo] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const navigate = useNavigate();

    const fetchVideo = useCallback(async () => {
        setIsLoading(true);
        setError(null);
        try {
            const response = await videoApiService.getVideoById(videoId);
            if (response.success) {
                setVideo(response.data);
            } else {
                throw new Error(response.message);
            }
        } catch (err) {
            setError(err.message);
            showToast.error(err.message || "Failed to load video details.");
        } finally {
            setIsLoading(false);
        }
    }, [videoId]);

    useEffect(() => {
        if (videoId) {
            fetchVideo();
        }
    }, [videoId, fetchVideo]);

    const updateVideo = async (updateData) => {
        try {
            await videoApiService.updateVideo(videoId, updateData);
            showToast.success("Video Data updated successfully.");
            fetchVideo();
        } catch (err) {
            showToast.error(err.message || "Update failed.");
            throw err;
        }
    };

    const deleteVideo = async () => {
        try {
            await videoApiService.deleteVideo(videoId);
            showToast.success("Video deleted successfully.");
            navigate("/dashboard");
        } catch (err) {
            showToast.error(err.message || "Deletion failed.");
            throw err;
        }
    };

    return { video, isLoading, error, fetchVideo, updateVideo, deleteVideo };
};
