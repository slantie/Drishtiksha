// src/hooks/useAuthQuery.js

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { authApi } from "../services/api/auth.api.js";
import { queryKeys } from "../lib/queryKeys.js";
import { showToast } from "../utils/toast.js";
import { authStorage } from "../utils/authStorage.js";

/**
 * Hook for the user login mutation.
 */
export const useLoginMutation = () => {
    const navigate = useNavigate();
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: ({ email, password }) =>
            authApi.login({ email, password }),
        onSuccess: (response) => {
            const { token, user } = response.data;

            // Store in sessionStorage (browser will handle session persistence)
            sessionStorage.setItem("authToken", token);
            sessionStorage.setItem("user", JSON.stringify(user));

            // Put the profile data into the cache for immediate UI updates
            queryClient.setQueryData(queryKeys.auth.profile(), response.data);

            // Optionally refetch authenticated queries (keeps cache consistent)
            queryClient.invalidateQueries(queryKeys.auth.profile());

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

/**
 * Hook for the user signup mutation.
 */
export const useSignupMutation = () => {
    return useMutation({
        mutationFn: authApi.signup,
        onSuccess: () => {
            showToast.success("Account created! Please log in to continue.");
        },
        onError: (error) => {
            showToast.error(
                error.message || "Signup failed. Please try again."
            );
        },
    });
};

/**
 * Hook for the user logout mutation.
 */
export const useLogoutMutation = () => {
    const queryClient = useQueryClient();
    const navigate = useNavigate();

    return useMutation({
        mutationFn: authApi.logout,
        onSuccess: () => {
            // Clear the entire query cache to remove all authenticated data.
            queryClient.clear();

            // Clear session storage
            sessionStorage.removeItem("authToken");
            sessionStorage.removeItem("user");

            showToast.success("You have been logged out.");
            navigate("/");
        },
        onError: (error) => {
            showToast.error(error.message || "Logout failed.");
        },
    });
};

/**
 * Hook to fetch the authenticated user's profile.
 */
export const useProfileQuery = () => {
    const token = sessionStorage.getItem("authToken");
    return useQuery({
        queryKey: queryKeys.auth.profile(),
        queryFn: authApi.getProfile,
        // Only run this query if a token exists.
        enabled: !!token,
        // The data is critical, so we treat it as fresh for a shorter time.
        staleTime: 1000 * 60, // 1 minute
        select: (response) => response.data,
    });
};

/**
 * Hook for the user profile update mutation.
 */
export const useUpdateProfileMutation = () => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: authApi.updateProfile,
        onSuccess: (updatedProfileResponse) => {
            // Update the profile query cache with the new data.
            queryClient.setQueryData(
                queryKeys.auth.profile(),
                updatedProfileResponse
            );
            showToast.success("Profile updated successfully!");
            // Refetch the profile query to get the latest data.
            queryClient.invalidateQueries(queryKeys.auth.profile());
        },
        onError: (error) => {
            showToast.error(error.message || "Profile update failed.");
        },
    });
};

/**
 * Hook for the user password update mutation.
 */
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

/**
 * Hook for the user avatar update mutation.
 */
export const useUpdateAvatarMutation = () => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: authApi.updateAvatar,
        onSuccess: () => {
            // Invalidate the profile query to refetch it with the new avatar URL.
            queryClient.invalidateQueries({
                queryKey: queryKeys.auth.profile(),
            });
            queryClient.invalidateQueries(queryKeys.auth.profile());
            showToast.success("Avatar updated successfully!");
        },
        onError: (error) => {
            showToast.error(error.message || "Avatar update failed.");
        },
    });
};

/**
 * Hook for the user avatar delete mutation.
 */
export const useDeleteAvatarMutation = () => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: authApi.deleteAvatar,
        onSuccess: () => {
            queryClient.invalidateQueries({
                queryKey: queryKeys.auth.profile(),
            });
            queryClient.invalidateQueries(queryKeys.auth.profile());
            showToast.success("Avatar removed.");
        },
        onError: (error) => {
            showToast.error(error.message || "Failed to remove avatar.");
        },
    });
};
