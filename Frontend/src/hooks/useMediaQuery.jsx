// src/hooks/useMediaQuery.jsx

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
// UPDATED: Importing the new mediaApi service
import { mediaApi } from "../services/api/media.api.js";
import { queryKeys } from "../lib/queryKeys.js";
import { showToast } from "../utils/toast.js";

/**
 * Hook to fetch the list of all media for the user.
 */
// RENAMED: from useVideosQuery to useMediaQuery
export const useMediaQuery = () => {
    return useQuery({
        // UPDATED: Using the new media query key
        queryKey: queryKeys.media.lists(),
        queryFn: mediaApi.getAll,
        select: (response) => response.data,
    });
};

/**
 * Hook to fetch a single media item by its ID, including all its analyses.
 */
// RENAMED: from useMediaQuery to useMediaItemQuery
export const useMediaItemQuery = (mediaId, options = {}) => {
    return useQuery({
        // UPDATED: Using the new media query key
        queryKey: queryKeys.media.detail(mediaId),
        queryFn: () => mediaApi.getById(mediaId),
        enabled: !!mediaId,
        select: (response) => response.data,
        ...options,
    });
};

/**
 * Hook for the media upload mutation.
 */
// RENAMED: from useUploadVideoMutation to useUploadMediaMutation
export const useUploadMediaMutation = () => {
    const queryClient = useQueryClient();
    const navigate = useNavigate();

    return useMutation({
        mutationFn: mediaApi.upload,
        onSuccess: (response) => {
            const newMedia = response.data;
            // UPDATED: Generic success message
            showToast.success("Upload complete! Analysis has been queued.");
            
            // Invalidate the media list to show the new item
            queryClient.invalidateQueries({ queryKey: queryKeys.media.lists() });

            // Navigate to the results page for the new media item
            navigate(`/results/${newMedia.id}`);
        },
        onError: (error) => {
            showToast.error(error.message || "Upload failed. Please try again.");
        },
    });
};


/**
 * Hook for the media metadata update mutation.
 */
// RENAMED: from useUpdateMediaMutation to useUpdateMediaMutation
export const useUpdateMediaMutation = () => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: ({ mediaId, updateData }) => mediaApi.update(mediaId, updateData),
        onSuccess: (_, { mediaId }) => {
            // UPDATED: Generic success message
            showToast.success("Media details updated.");
            queryClient.invalidateQueries({ queryKey: queryKeys.media.lists() });
            queryClient.invalidateQueries({ queryKey: queryKeys.media.detail(mediaId) });
        },
        onError: (error) => {
            showToast.error(error.message || "Failed to update media details.");
        },
    });
};

/**
 * Hook for the media deletion mutation.
 */
export const useDeleteMediaMutation = () => {
    const queryClient = useQueryClient();
    const navigate = useNavigate();
    return useMutation({
        mutationFn: mediaApi.delete,
        onSuccess: (_, mediaId) => {
            // UPDATED: Generic success message
            showToast.success("Media deleted successfully.");
            queryClient.removeQueries({ queryKey: queryKeys.media.detail(mediaId) });
            queryClient.invalidateQueries({ queryKey: queryKeys.media.lists() });
            navigate("/dashboard");
        },
        onError: (error) => {
            showToast.error(error.message || "Failed to delete media.");
        },
    });
};

/**
 * Hook for triggering a manual analysis run.
 */
export const useCreateAnalysisMutation = () => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: ({ mediaId, analysisConfig }) => mediaApi.createAnalysis(mediaId, analysisConfig),
        onSuccess: (data, { mediaId }) => {
            showToast.info("New analysis has been queued.");
            queryClient.invalidateQueries({ queryKey: queryKeys.media.detail(mediaId) });
        },
        onError: (error) => {
            showToast.error(error.message || "Failed to start new analysis.");
        },
    });
};

/**
 * Hook that efficiently computes media statistics from cached data.
 */
// RENAMED: from useVideoStats to useMediaStats
export const useMediaStats = () => {
    // UPDATED: Using the new useMediaQuery hook
    const { data: mediaItems = [], isLoading, error } = useMediaQuery();
    const stats = useMemo(() => {
        if (!Array.isArray(mediaItems)) {
            return { total: 0, analyzed: 0, realDetections: 0, fakeDetections: 0, totalAnalyses: 0 };
        }
        return mediaItems.reduce(
            (acc, media) => {
                acc.total++;
                if (["ANALYZED", "PARTIALLY_ANALYZED"].includes(media.status)) {
                    acc.analyzed++;
                }
                const completedAnalyses = media.analyses?.filter((a) => a.status === "COMPLETED") || [];
                acc.totalAnalyses += completedAnalyses.length;
                completedAnalyses.forEach((analysis) => {
                    if (analysis.prediction === "REAL") acc.realDetections++;
                    if (analysis.prediction === "FAKE") acc.fakeDetections++;
                });
                return acc;
            },
            { total: 0, analyzed: 0, realDetections: 0, fakeDetections: 0, totalAnalyses: 0 }
        );
    }, [mediaItems]);

    return { stats, isLoading, error };
};