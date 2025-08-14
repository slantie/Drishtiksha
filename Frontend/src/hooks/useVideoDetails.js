// src/hooks/useVideoDetails.js
// ⚠️ DEPRECATED: This hook is deprecated. Use hooks from useVideosQuery.js instead.

import { useNavigate } from "react-router-dom";
import {
    useVideoQuery,
    useUpdateVideoMutation,
    useDeleteVideoMutation,
} from "./useVideosQuery.js";

/**
 * @deprecated Use individual hooks from useVideosQuery.js instead:
 * - useVideoQuery(videoId) for fetching a single video
 * - useUpdateVideoMutation() for updating
 * - useDeleteVideoMutation() for deleting
 *
 * This legacy wrapper is kept for compatibility but will be removed.
 */
export const useVideoDetails = (videoId) => {
    console.warn(
        "⚠️ useVideoDetails hook is deprecated. Please use useVideoQuery, useUpdateVideoMutation, " +
            "and useDeleteVideoMutation from useVideosQuery.js instead. See TANSTACK_MIGRATION.md for migration guide."
    );

    const navigate = useNavigate();

    // Use TanStack Query hooks
    const {
        data: video,
        isLoading,
        error,
        refetch: fetchVideo,
    } = useVideoQuery(videoId);
    const updateMutation = useUpdateVideoMutation();
    const deleteMutation = useDeleteVideoMutation();

    const updateVideo = async (updateData) => {
        return updateMutation.mutateAsync({ videoId, updateData });
    };

    const deleteVideo = async () => {
        await deleteMutation.mutateAsync(videoId);
        navigate("/dashboard");
    };

    return {
        video,
        isLoading:
            isLoading || updateMutation.isPending || deleteMutation.isPending,
        error: error || updateMutation.error || deleteMutation.error,
        fetchVideo,
        updateVideo,
        deleteVideo,
    };
};
