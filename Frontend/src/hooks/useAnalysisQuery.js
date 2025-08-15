// src/hooks/useAnalysisQuery.js

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { videoApi } from "../services/api/video.api.js";
import { queryKeys } from "../lib/queryKeys.js";
import { showToast } from "../utils/toast.js";

/**
 * Hook to get model service status
 */
export const useModelStatusQuery = () => {
    return useQuery({
        queryKey: queryKeys.analysis.modelStatus(),
        queryFn: videoApi.getModelStatus,
        select: (data) => data?.data || data,
        staleTime: 5 * 60 * 1000, // 5 minutes
        refetchInterval: 30 * 1000, // Refetch every 30 seconds
    });
};

/**
 * Hook to get analysis results for a video
 */
export const useAnalysisResultsQuery = (videoId, filters = {}) => {
    return useQuery({
        queryKey: queryKeys.analysis.results(videoId, filters),
        queryFn: () => videoApi.getAnalysisResults(videoId, filters),
        enabled: !!videoId,
        select: (data) => data?.data || data,
    });
};

/**
 * Hook to get specific analysis by type and model
 */
export const useSpecificAnalysisQuery = (videoId, type, model) => {
    return useQuery({
        queryKey: queryKeys.analysis.specific(videoId, type, model),
        queryFn: () => videoApi.getSpecificAnalysis(videoId, type, model),
        enabled: !!(videoId && type && model),
        select: (data) => data?.data || data,
    });
};

/**
 * Hook for creating a specific analysis
 */
export const useCreateAnalysisMutation = () => {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: ({ videoId, type, model }) =>
            videoApi.createAnalysis(videoId, type, model),
        onMutate: ({ videoId, type, model }) => {
            showToast.info(`Starting ${type} analysis with ${model}...`);
        },
        onSuccess: (data, { videoId, type, model }) => {
            // Invalidate relevant queries
            queryClient.invalidateQueries({
                queryKey: queryKeys.analysis.results(videoId),
            });
            queryClient.invalidateQueries({
                queryKey: queryKeys.videos.detail(videoId),
            });
            queryClient.invalidateQueries({
                queryKey: queryKeys.videos.lists(),
            });

            showToast.success(`${type} analysis completed successfully!`);
        },
        onError: (error, { type, model }) => {
            showToast.error(
                error.message ||
                    `Failed to complete ${type} analysis with ${model}`
            );
        },
    });
};

/**
 * Hook for creating multiple analyses (batch)
 */
export const useCreateMultipleAnalysesMutation = () => {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: ({ videoId, analysisConfigs }) =>
            videoApi.createMultipleAnalyses(videoId, analysisConfigs),
        onMutate: ({ analysisConfigs }) => {
            showToast.info(`Starting ${analysisConfigs.length} analyses...`);
        },
        onSuccess: (data, { videoId, analysisConfigs }) => {
            // Invalidate relevant queries
            queryClient.invalidateQueries({
                queryKey: queryKeys.analysis.results(videoId),
            });
            queryClient.invalidateQueries({
                queryKey: queryKeys.videos.detail(videoId),
            });
            queryClient.invalidateQueries({
                queryKey: queryKeys.videos.lists(),
            });

            showToast.success(
                `All ${analysisConfigs.length} analyses completed!`
            );
        },
        onError: (error, { analysisConfigs }) => {
            showToast.error(
                error.message || `Failed to complete batch analysis`
            );
        },
    });
};

/**
 * Hook for creating model-specific visualization
 */
export const useCreateModelVisualizationMutation = () => {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: ({ videoId, model }) =>
            videoApi.createModelVisualization(videoId, model),
        onMutate: ({ model }) => {
            showToast.info(
                `Generating visualization with ${model}... This may take several minutes.`
            );
        },
        onSuccess: (data, { videoId, model }) => {
            const updatedVideo = data?.data || data;

            // Invalidate relevant queries
            queryClient.invalidateQueries({
                queryKey: queryKeys.analysis.results(videoId),
            });
            queryClient.invalidateQueries({
                queryKey: queryKeys.videos.detail(videoId),
            });
            queryClient.invalidateQueries({
                queryKey: queryKeys.videos.lists(),
            });

            showToast.success(
                `Visualization with ${model} generated successfully!`
            );
        },
        onError: (error, { model }) => {
            showToast.error(
                error.message ||
                    `Failed to generate visualization with ${model}`
            );
        },
    });
};

/**
 * Hook that provides comprehensive analysis operations for a video
 */
export const useVideoAnalysis = (videoId) => {
    const modelStatusQuery = useModelStatusQuery();
    const analysisResultsQuery = useAnalysisResultsQuery(videoId);
    const createAnalysisMutation = useCreateAnalysisMutation();
    const createMultipleAnalysesMutation = useCreateMultipleAnalysesMutation();
    const createVisualizationMutation = useCreateModelVisualizationMutation();

    const createAnalysis = (type, model) => {
        return createAnalysisMutation.mutateAsync({ videoId, type, model });
    };

    const createMultipleAnalyses = (analysisConfigs) => {
        return createMultipleAnalysesMutation.mutateAsync({
            videoId,
            analysisConfigs,
        });
    };

    const createVisualization = (model) => {
        return createVisualizationMutation.mutateAsync({ videoId, model });
    };

    const isLoading =
        analysisResultsQuery.isLoading ||
        createAnalysisMutation.isPending ||
        createMultipleAnalysesMutation.isPending ||
        createVisualizationMutation.isPending;

    const error =
        analysisResultsQuery.error ||
        createAnalysisMutation.error ||
        createMultipleAnalysesMutation.error ||
        createVisualizationMutation.error;

    return {
        // Data
        modelStatus: modelStatusQuery.data,
        analysisResults: analysisResultsQuery.data,

        // Loading states
        isLoading,
        isCreatingAnalysis: createAnalysisMutation.isPending,
        isCreatingMultiple: createMultipleAnalysesMutation.isPending,
        isCreatingVisualization: createVisualizationMutation.isPending,
        isModelStatusLoading: modelStatusQuery.isLoading,

        // Error states
        error,
        modelStatusError: modelStatusQuery.error,

        // Actions
        createAnalysis,
        createMultipleAnalyses,
        createVisualization,
        refetchAnalysis: analysisResultsQuery.refetch,
        refetchModelStatus: modelStatusQuery.refetch,
    };
};
