// src/hooks/useVideos.js

import { useState, useEffect, useCallback, useMemo } from "react";
import { videoApiService } from "../services/videoApiService.js";
import { showToast } from "../utils/toast.js";

export const useVideos = () => {
    const [videos, setVideos] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);

    const fetchVideos = useCallback(async () => {
        try {
            setIsLoading(true);
            const response = await videoApiService.getAllVideos();
            if (response.success) {
                setVideos(response.data);
                console.log("Videos Data fetched successfully!", response.data);
                // showToast.success("Videos Data fetched successfully!");
            } else {
                throw new Error(response.message || "Failed to fetch videos");
            }
        } catch (err) {
            setError(err.message);
            showToast.error(err.message);
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchVideos();
    }, [fetchVideos]);

    const uploadVideo = async (formData) => {
        try {
            await videoApiService.uploadVideo(formData);
            showToast.success("Video uploaded!");
            showToast.info("Video Analysis has started.");
            fetchVideos();
        } catch (err) {
            showToast.error(err.message || "Upload failed.");
            throw err;
        }
    };

    const updateVideo = async (videoId, updateData) => {
        try {
            await videoApiService.updateVideo(videoId, updateData);
            showToast.success("Video Data updated successfully.");
            fetchVideos();
        } catch (err) {
            showToast.error(err.message || "Update failed.");
            throw err;
        }
    };

    const deleteVideo = async (videoId) => {
        try {
            await videoApiService.deleteVideo(videoId);
            showToast.success("Video deleted successfully.");
            fetchVideos();
        } catch (err) {
            showToast.error(err.message || "Deletion failed.");
            throw err;
        }
    };

    const stats = useMemo(() => {
        return videos.reduce(
            (acc, v) => {
                acc.total++;
                if (v.status === "ANALYZED") acc.analyzed++;
                v.analyses.forEach((a) => {
                    if (a.prediction === "REAL") acc.realDetections++;
                    if (a.prediction === "FAKE") acc.fakeDetections++;
                });
                return acc;
            },
            { total: 0, analyzed: 0, realDetections: 0, fakeDetections: 0 }
        );
    }, [videos]);

    return {
        videos,
        isLoading,
        error,
        stats,
        fetchVideos,
        uploadVideo,
        updateVideo,
        deleteVideo,
    };
};
