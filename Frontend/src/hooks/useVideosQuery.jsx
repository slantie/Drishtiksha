// src/hooks/useVideosQuery.jsx

import React from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { videoApi } from "../services/api/video.api.js";
import { queryKeys } from "../lib/queryKeys.js";
import { showToast } from "../utils/toast.js";
import { toastManager } from "../lib/toastManager.js";
import { ToastProgress } from "../components/ui/ToastProgress.jsx";

/**
 * Hook to fetch the list of all videos for the user.
 */
export const useVideosQuery = () => {
    return useQuery({
        queryKey: queryKeys.videos.lists(),
        queryFn: videoApi.getAll,
        select: (response) => response.data,
    });
};

/**
 * Hook to fetch a single video by its ID, including all its analyses.
 */
export const useVideoQuery = (videoId, options = {}) => {
    return useQuery({
        queryKey: queryKeys.videos.detail(videoId),
        queryFn: () => videoApi.getById(videoId),
        enabled: !!videoId,
        select: (response) => response.data,
        ...options,
    });
};

/**
 * Hook for the video upload mutation.
 */
export const useUploadVideoMutation = () => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: videoApi.upload,
        onSuccess: (response) => {
            const newVideo = response.data;
            showToast.success("Upload complete! Analysis has been queued.");

            // The first socket event will now create the toast, but we can log here for debugging.
            console.log(
                `[Upload] Video ${newVideo.id} uploaded. Awaiting 'PROCESSING_STARTED' socket event to create progress toast.`
            );

            queryClient.invalidateQueries({
                queryKey: queryKeys.videos.lists(),
            });
        },
        onError: (error) => {
            showToast.error(
                error.message || "Upload failed. Please try again."
            );
        },
    });
};

/**
 * Hook for the video metadata update mutation.
 */
export const useUpdateVideoMutation = () => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: ({ videoId, updateData }) =>
            videoApi.update(videoId, updateData),
        onSuccess: (_, { videoId }) => {
            showToast.success("Video details updated.");
            queryClient.invalidateQueries({
                queryKey: queryKeys.videos.lists(),
            });
            queryClient.invalidateQueries({
                queryKey: queryKeys.videos.detail(videoId),
            });
        },
        onError: (error) => {
            showToast.error(error.message || "Failed to update video.");
        },
    });
};

/**
 * Hook for the video deletion mutation.
 */
export const useDeleteVideoMutation = () => {
    const queryClient = useQueryClient();
    const navigate = useNavigate();
    return useMutation({
        mutationFn: videoApi.delete,
        onSuccess: (_, videoId) => {
            showToast.success("Video deleted successfully.");
            queryClient.removeQueries({
                queryKey: queryKeys.videos.detail(videoId),
            });
            queryClient.invalidateQueries({
                queryKey: queryKeys.videos.lists(),
            });
            navigate("/dashboard");
        },
        onError: (error) => {
            showToast.error(error.message || "Failed to delete video.");
        },
    });
};

/**
 * Hook for triggering a manual analysis run.
 */
export const useCreateAnalysisMutation = () => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: ({ videoId, analysisConfig }) =>
            videoApi.createAnalysis(videoId, analysisConfig),
        onSuccess: (data, { videoId }) => {
            showToast.success("Analysis Versioning Coming Soon!");
            queryClient.invalidateQueries({
                queryKey: queryKeys.videos.detail(videoId),
            });
        },
        onError: (error) => {
            showToast.error(error.message || "Failed to start new analysis.");
        },
    });
};

/**
 * Hook that efficiently computes video statistics from cached data.
 */
export const useVideoStats = () => {
    const { data: videos = [], isLoading, error } = useVideosQuery();
    const stats = useMemo(() => {
        if (!Array.isArray(videos)) {
            return {
                total: 0,
                analyzed: 0,
                realDetections: 0,
                fakeDetections: 0,
                totalAnalyses: 0,
            };
        }
        return videos.reduce(
            (acc, video) => {
                acc.total++;
                if (
                    video.status === "ANALYZED" ||
                    video.status === "PARTIALLY_ANALYZED"
                ) {
                    acc.analyzed++;
                }
                const completedAnalyses =
                    video.analyses?.filter((a) => a.status === "COMPLETED") ||
                    [];
                acc.totalAnalyses += completedAnalyses.length;
                completedAnalyses.forEach((analysis) => {
                    if (analysis.prediction === "REAL") acc.realDetections++;
                    if (analysis.prediction === "FAKE") acc.fakeDetections++;
                });
                return acc;
            },
            {
                total: 0,
                analyzed: 0,
                realDetections: 0,
                fakeDetections: 0,
                totalAnalyses: 0,
            }
        );
    }, [videos]);

    return { stats, isLoading, error };
};
