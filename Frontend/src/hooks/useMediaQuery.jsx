// src/hooks/useMediaQuery.jsx

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { mediaApi } from "../services/api/media.api.js";
import { queryKeys } from "../lib/queryKeys.js";
import { showToast } from "../utils/toast.jsx";
import { socketService } from "../lib/socket.jsx";

/**
 * ✨ NEW: Hook to check WebSocket connection status.
 * Used for smart polling logic.
 */
export const useSocketStatus = () => {
  const [isConnected, setIsConnected] = useState(
    socketService.isConnected()
  );

  useEffect(() => {
    // Check socket status every second
    const interval = setInterval(() => {
      setIsConnected(socketService.isConnected());
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return { isConnected };
};

export const useMediaQuery = () => {
  return useQuery({
    queryKey: queryKeys.media.lists(),
    queryFn: mediaApi.getAll,
    select: (response) => response.data,
  });
};

export const useMediaItemQuery = (mediaId, options = {}) => {
  const { isConnected: isSocketConnected } = useSocketStatus();

  return useQuery({
    queryKey: queryKeys.media.detail(mediaId),
    queryFn: () => mediaApi.getById(mediaId),
    enabled: !!mediaId,
    select: (response) => response.data,
    staleTime: 5000, // Consider data stale after 5 seconds
    
    // ✨ NEW: Smart polling based on WebSocket status and media state
    refetchInterval: (query) => {
      const media = query.state.data;
      const status = media?.analysisRuns?.[0]?.status || media?.status;

      // Stop polling if not in processing state
      if (!status || !["QUEUED", "PROCESSING"].includes(status)) {
        return false;
      }

      // Smart polling strategy:
      // - If WebSocket is connected: poll slowly as fallback (30 seconds)
      //   WebSocket updates should handle most updates in real-time
      // - If WebSocket is disconnected: poll actively (5 seconds)
      //   Fall back to polling for updates
      return isSocketConnected ? 30000 : 5000;
    },
    
    ...options,
  });
};

export const useUploadMediaMutation = () => {
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  return useMutation({
    mutationFn: mediaApi.upload,
    onSuccess: (response) => {
      const newMedia = response.data;
      showToast.success(
        `Upload complete! "${newMedia.filename}" queued for analysis.`
      );
      queryClient.invalidateQueries({ queryKey: queryKeys.media.lists() });
      navigate(`/results/${newMedia.id}`);
    },
    onError: (error) => {
      showToast.error(error.message || "Upload failed. Please try again.");
    },
  });
};

export const useUpdateMediaMutation = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ mediaId, updateData }) =>
      mediaApi.update(mediaId, updateData),
    onSuccess: (response, { mediaId }) => {
      // response contains the updated media object
      showToast.success(`Media "${response.data.filename || "item"}" updated.`);
      queryClient.invalidateQueries({ queryKey: queryKeys.media.lists() });
      queryClient.invalidateQueries({
        queryKey: queryKeys.media.detail(mediaId),
      });
    },
    onError: (error) => {
      showToast.error(error.message || "Failed to update media details.");
    },
  });
};

export const useDeleteMediaMutation = () => {
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  return useMutation({
    mutationFn: mediaApi.delete,
    onSuccess: (_, mediaId) => {
      // No data in response from delete
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

export const useRerunAnalysisMutation = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (mediaId) => mediaApi.rerunAnalysis(mediaId),
    onSuccess: (response, mediaId) => {
      // response contains the updated media with new run
      showToast.info(
        `New analysis run for "${
          response.data.filename || "media"
        }" has been queued.`
      );
      queryClient.invalidateQueries({
        queryKey: queryKeys.media.detail(mediaId),
      });
    },
    onError: (error) => {
      showToast.error(error.message || "Failed to start new analysis.");
    },
  });
};

export const useMediaStats = () => {
  const { data: mediaItems = [], isLoading, error } = useMediaQuery();

  const stats = useMemo(() => {
    if (!Array.isArray(mediaItems)) {
      return {
        total: 0,
        realDetections: 0,
        fakeDetections: 0,
        totalAnalyses: 0,
        processing: 0,
        analyzed: 0,
        failed: 0,
      };
    }
    return mediaItems.reduce(
      (acc, media) => {
        acc.total++;
        const latestRun = media.analysisRuns?.[0]; // Runs are sorted desc by runNumber

        // Update status counts based on overall media status
        if (media.status === "PROCESSING" || media.status === "QUEUED")
          acc.processing++;
        else if (media.status === "ANALYZED") acc.analyzed++;
        else if (media.status === "FAILED") acc.failed++;

        if (latestRun) {
          const completedAnalyses =
            latestRun.analyses?.filter((a) => a.status === "COMPLETED") || [];
          acc.totalAnalyses += completedAnalyses.length;
          completedAnalyses.forEach((analysis) => {
            if (analysis.prediction === "REAL") acc.realDetections++;
            if (analysis.prediction === "FAKE") acc.fakeDetections++;
          });
        }
        return acc;
      },
      {
        total: 0,
        realDetections: 0,
        fakeDetections: 0,
        totalAnalyses: 0,
        processing: 0,
        analyzed: 0,
        failed: 0,
      }
    );
  }, [mediaItems]);

  return { stats, isLoading, error };
};
