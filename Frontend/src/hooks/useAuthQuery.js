// src/hooks/useAuthQuery.js

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { authApi } from "../services/api/auth.api.js";
import { queryKeys } from "../lib/queryKeys.js";
import { showToast } from "../utils/toast.jsx";
import { authStorage } from "../utils/authStorage.js";

export const useLoginMutation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: authApi.login,
    // onSuccess now only invalidates relevant queries, actual storage/navigation is handled in AuthContext
    onSuccess: (response) => {
      // The AuthContext's login function handles setting authStorage, localToken, and navigation.
      // This hook's onSuccess will primarily ensure other data (like media lists) are refreshed.
      queryClient.invalidateQueries({ queryKey: queryKeys.media.lists() });
      queryClient.invalidateQueries({ queryKey: queryKeys.monitoring.all });
    },
    onError: (error) => {
      showToast.error(
        error.message || "Login failed. Please check your credentials."
      );
    },
  });
};

export const useSignupMutation = () => {
  return useMutation({
    mutationFn: authApi.signup,
    // onSuccess now only shows a toast, navigation is handled in AuthContext
    // The AuthContext's signup function handles showing the success toast and navigation.
    onError: (error) => {
      showToast.error(error.message || "Signup failed. Please try again.");
    },
  });
};

export const useLogoutMutation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: authApi.logout,
    // onSuccess is removed here because AuthContext's logout function
    // already handles clearing storage, cache, and navigation after calling this mutation.
    // This prevents redundant actions and centralizes logic.
    onSuccess: () => {
      // No action here, AuthContext's logout wrapper handles everything.
    },
    onError: (error) => {
      // Even if the backend logout fails, we still want to clear client-side state.
      // AuthContext will handle this by checking for local token presence.
      showToast.error(
        error.message ||
          "Logout failed, but your local session has been cleared."
      );
    },
  });
};

export const useProfileQuery = (token) => {
  // Accept token as prop
  return useQuery({
    queryKey: queryKeys.auth.profile(),
    queryFn: authApi.getProfile,
    enabled: !!token, // Only run this query if a token exists
    staleTime: 1000 * 60 * 5, // 5 minutes
    select: (response) => response.data,
  });
};

export const useUpdateProfileMutation = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: authApi.updateProfile,
    onSuccess: (updatedProfileResponse) => {
      queryClient.setQueryData(
        queryKeys.auth.profile(),
        updatedProfileResponse.data
      );
      showToast.success("Profile updated successfully!");
    },
    onError: (error) => {
      showToast.error(error.message || "Profile update failed.");
    },
  });
};

export const useUpdatePasswordMutation = () => {
  return useMutation({
    mutationFn: authApi.updatePassword,
    onSuccess: () => {
      showToast.success("Password updated successfully!");
    },
    onError: (error) => {
      showToast.error(error.message || "Password update failed.");
    },
  });
};
