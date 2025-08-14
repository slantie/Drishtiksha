// src/hooks/useVideos.js
// ⚠️ DEPRECATED: This hook is deprecated. Use hooks from useVideosQuery.js instead.
// This hook is kept for backward compatibility but will be removed in a future version.

import {
    useVideosQuery,
    useUploadVideoMutation,
    useUpdateVideoMutation,
    useDeleteVideoMutation,
    useVideoStats,
} from "./useVideosQuery.js";

/**
 * @deprecated Use individual hooks from useVideosQuery.js instead:
 * - useVideosQuery() for fetching videos
 * - useUploadVideoMutation() for uploading
 * - useUpdateVideoMutation() for updating
 * - useDeleteVideoMutation() for deleting
 * - useVideoStats() for statistics
 *
 * This legacy wrapper is kept for compatibility but will be removed.
 */
export const useVideos = () => {
    console.warn(
        "⚠️ useVideos hook is deprecated. Please use individual hooks from useVideosQuery.js instead. " +
            "See TANSTACK_MIGRATION.md for migration guide."
    );

    const {
        data: videos = [],
        isLoading,
        error,
        refetch: fetchVideos,
    } = useVideosQuery();
    const uploadMutation = useUploadVideoMutation();
    const updateMutation = useUpdateVideoMutation();
    const deleteMutation = useDeleteVideoMutation();
    const { stats } = useVideoStats();

    const uploadVideo = async (formData) => {
        return uploadMutation.mutateAsync(formData);
    };

    const updateVideo = async (videoId, updateData) => {
        return updateMutation.mutateAsync({ videoId, updateData });
    };

    const deleteVideo = async (videoId) => {
        return deleteMutation.mutateAsync(videoId);
    };

    return {
        videos,
        isLoading:
            isLoading ||
            uploadMutation.isPending ||
            updateMutation.isPending ||
            deleteMutation.isPending,
        error:
            error ||
            uploadMutation.error ||
            updateMutation.error ||
            deleteMutation.error,
        stats,
        fetchVideos,
        uploadVideo,
        updateVideo,
        deleteVideo,
    };
};
