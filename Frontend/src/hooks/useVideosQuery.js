// src/hooks/useVideosQuery.js

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { videoApi } from "../services/api/video.api.js";
import { queryKeys } from "../lib/queryKeys.js";
import { showToast } from "../utils/toast.js";
import { useMemo } from "react";

/**
 * Hook to fetch all videos with TanStack Query
 */
export const useVideosQuery = () => {
    return useQuery({
        queryKey: queryKeys.videos.lists(),
        queryFn: videoApi.getAllVideos,
        select: (data) => data?.data || data, // Handle different response structures
    });
};

/**
 * Hook to fetch a single video by ID
 */
export const useVideoQuery = (videoId) => {
    return useQuery({
        queryKey: queryKeys.videos.detail(videoId),
        queryFn: () => videoApi.getVideoById(videoId),
        enabled: !!videoId, // Only run if videoId exists
        select: (data) => data?.data || data,
    });
};

/**
 * Hook for video upload mutation
 */
export const useUploadVideoMutation = () => {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: videoApi.uploadVideo,
        onSuccess: () => {
            // Invalidate and refetch videos list
            queryClient.invalidateQueries({
                queryKey: queryKeys.videos.lists(),
            });
            showToast.success("Video uploaded successfully!");
            showToast.info("Video analysis has started.");
        },
        onError: (error) => {
            showToast.error(error.message || "Upload failed.");
        },
    });
};

/**
 * Hook for video update mutation
 */
export const useUpdateVideoMutation = () => {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: ({ videoId, updateData }) =>
            videoApi.updateVideo(videoId, updateData),
        onSuccess: (_, variables) => {
            // Invalidate both the video list and the specific video
            queryClient.invalidateQueries({
                queryKey: queryKeys.videos.lists(),
            });
            queryClient.invalidateQueries({
                queryKey: queryKeys.videos.detail(variables.videoId),
            });
            showToast.success("Video updated successfully!");
        },
        onError: (error) => {
            showToast.error(error.message || "Update failed.");
        },
    });
};

/**
 * Hook for video deletion mutation
 */
export const useDeleteVideoMutation = () => {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: videoApi.deleteVideo,
        onSuccess: (_, videoId) => {
            // Remove the deleted video from cache and invalidate list
            queryClient.removeQueries({
                queryKey: queryKeys.videos.detail(videoId),
            });
            queryClient.invalidateQueries({
                queryKey: queryKeys.videos.lists(),
            });
            showToast.success("Video deleted successfully!");
        },
        onError: (error) => {
            showToast.error(error.message || "Deletion failed.");
        },
    });
};

/**
 * Hook that provides video statistics
 */
export const useVideoStats = () => {
    const { data: videos = [], isLoading, error } = useVideosQuery();

    const stats = useMemo(() => {
        if (!videos || !Array.isArray(videos)) {
            return {
                total: 0,
                analyzed: 0,
                realDetections: 0,
                fakeDetections: 0,
            };
        }

        return videos.reduce(
            (acc, video) => {
                acc.total++;
                if (video.status === "ANALYZED") acc.analyzed++;

                if (video.analyses && Array.isArray(video.analyses)) {
                    video.analyses.forEach((analysis) => {
                        if (analysis.prediction === "REAL")
                            acc.realDetections++;
                        if (analysis.prediction === "FAKE")
                            acc.fakeDetections++;
                    });
                }

                return acc;
            },
            { total: 0, analyzed: 0, realDetections: 0, fakeDetections: 0 }
        );
    }, [videos]);

    return {
        stats,
        isLoading,
        error,
    };
};

/**
 * Hook for visual analysis generation mutation
 */
export const useVisualAnalysisMutation = () => {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: videoApi.generateVisualAnalysis, // The API function to call
        onSuccess: (data) => {
            const updatedVideo = data?.data || data;
            const videoId = updatedVideo?.id;

            if (!videoId) return;

            // When the mutation is successful, invalidate the queries
            // for the main video list and the specific video detail.
            // This will trigger a refetch and update the UI with the new `visualizedUrl`.
            queryClient.invalidateQueries({
                queryKey: queryKeys.videos.lists(),
            });
            queryClient.invalidateQueries({
                queryKey: queryKeys.videos.detail(videoId),
            });
            showToast.success("Visual analysis generated successfully!");
        },
        onError: (error) => {
            showToast.error(error.message || "Visual analysis generation failed.");
        },
        onMutate: () => {
            // Optional: Show a toast when the process starts
            showToast.info("Generating visual analysis... This may take several minutes.");
        },
    });
};