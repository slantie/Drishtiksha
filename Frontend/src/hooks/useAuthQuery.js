// src/hooks/useAuthQuery.js

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { authApi } from "../services/api/auth.api.js";
import { queryKeys } from "../lib/queryKeys.js";
import { showToast } from "../utils/toast.js";
import { authStorage } from "../utils/authStorage.js";

export const useLoginMutation = () => {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: authApi.login,
    onSuccess: (response) => {
      const { token, user } = response.data;
      authStorage.set({ token, user, rememberMe: true });
      queryClient.setQueryData(queryKeys.auth.profile(), response.data.user);
      showToast.success("Login successful!");
      navigate("/dashboard");
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
    onSuccess: () => {
      showToast.success("Account created! Please log in to continue.");
    },
    onError: (error) => {
      showToast.error(error.message || "Signup failed. Please try again.");
    },
  });
};

export const useLogoutMutation = () => {
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  return useMutation({
    mutationFn: authApi.logout,
    onSuccess: () => {
      authStorage.clear();
      queryClient.clear();
      showToast.success("You have been logged out.");
      navigate("/auth");
    },
    onError: (error) => {
      showToast.error(error.message || "Logout failed.");
    },
  });
};

export const useProfileQuery = () => {
  const { token } = authStorage.get();
  return useQuery({
    queryKey: queryKeys.auth.profile(),
    queryFn: authApi.getProfile,
    enabled: !!token,
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
