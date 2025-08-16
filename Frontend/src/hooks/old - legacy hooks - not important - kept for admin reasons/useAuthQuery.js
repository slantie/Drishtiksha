// src/hooks/useAuthQuery.js

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { authApi } from "../../services/api/auth.api.js";
import { queryKeys } from "../../lib/queryKeys.js";
import { showToast } from "../../utils/toast.js";
import { useNavigate } from "react-router-dom";

/**
 * Hook for login mutation
 */
export const useLoginMutation = () => {
    const navigate = useNavigate();

    return useMutation({
        mutationFn: ({ email, password }) => authApi.login({ email, password }),
        onSuccess: (data, { rememberMe }) => {
            const storage = rememberMe ? localStorage : sessionStorage;

            // Store token and user data
            storage.setItem("authToken", data.data.token);
            storage.setItem("user", JSON.stringify(data.data.user));

            showToast.success("Login success!");

            // Navigate to dashboard
            navigate("/dashboard");
        },
        onError: (error) => {
            showToast.error(error.message || "Login failed");
        },
    });
};

/**
 * Hook for signup mutation
 */
export const useSignupMutation = () => {
    return useMutation({
        mutationFn: authApi.signup,
        onSuccess: () => {
            showToast.success("Account created successfully! Please login.");
        },
        onError: (error) => {
            showToast.error(error.message || "Signup failed");
        },
    });
};

/**
 * Hook for logout mutation
 */
export const useLogoutMutation = () => {
    const queryClient = useQueryClient();
    const navigate = useNavigate();

    return useMutation({
        mutationFn: authApi.logout,
        onSuccess: () => {
            // Clear all queries from cache
            queryClient.clear();

            showToast.success("Logout success!");

            // Navigate to auth page
            navigate("/auth");
        },
        onError: (error) => {
            showToast.error(error.message || "Logout failed");
        },
    });
};

/**
 * Hook to get user profile
 */
export const useProfileQuery = () => {
    return useQuery({
        queryKey: queryKeys.auth.profile(),
        queryFn: authApi.getProfile,
        enabled: !!(
            localStorage.getItem("authToken") ||
            sessionStorage.getItem("authToken")
        ),
        select: (data) => data?.data || data,
        retry: false, // Don't retry profile requests to avoid spam
    });
};

/**
 * Hook for profile update mutation
 */
export const useUpdateProfileMutation = () => {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: authApi.updateProfile,
        onSuccess: (data) => {
            // Update the profile in cache
            queryClient.setQueryData(queryKeys.auth.profile(), data);

            // Also update localStorage/sessionStorage
            const storage = localStorage.getItem("user")
                ? localStorage
                : sessionStorage;
            storage.setItem("user", JSON.stringify(data.data || data));

            showToast.success("Profile updated successfully!");
        },
        onError: (error) => {
            showToast.error(error.message || "Profile update failed");
        },
    });
};

/**
 * Hook for password update mutation
 */
export const useUpdatePasswordMutation = () => {
    return useMutation({
        mutationFn: authApi.updatePassword,
        onSuccess: () => {
            showToast.success("Password updated successfully!");
        },
        onError: (error) => {
            showToast.error(error.message || "Password update failed");
        },
    });
};

/**
 * Hook for avatar update mutation
 */
export const useUpdateAvatarMutation = () => {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: authApi.updateAvatar,
        onSuccess: () => {
            // Invalidate profile to refetch with new avatar
            queryClient.invalidateQueries({
                queryKey: queryKeys.auth.profile(),
            });
            showToast.success("Avatar updated successfully!");
        },
        onError: (error) => {
            showToast.error(error.message || "Avatar update failed");
        },
    });
};

/**
 * Hook for avatar deletion mutation
 */
export const useDeleteAvatarMutation = () => {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: authApi.deleteAvatar,
        onSuccess: () => {
            // Invalidate profile to refetch without avatar
            queryClient.invalidateQueries({
                queryKey: queryKeys.auth.profile(),
            });
            showToast.success("Avatar deleted successfully!");
        },
        onError: (error) => {
            showToast.error(error.message || "Avatar deletion failed");
        },
    });
};
