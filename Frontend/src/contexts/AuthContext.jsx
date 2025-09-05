// src/contexts/AuthContext.jsx

import React, {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
} from "react";
import {
  useLoginMutation,
  useSignupMutation,
  useLogoutMutation,
  useProfileQuery,
} from "../hooks/useAuthQuery.js";
import { socketService } from "../lib/socket.jsx";
import { showToast } from "../utils/toast.jsx";
import { authStorage } from "../utils/authStorage.js";
import { useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom"; // Import useNavigate

export const AuthContext = createContext(null);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used within an AuthProvider");
  return context;
};

export const AuthProvider = ({ children }) => {
  const queryClient = useQueryClient();
  const navigate = useNavigate(); // Initialize useNavigate

  // Manage local token state which will drive socket connection
  const [localToken, setLocalToken] = useState(authStorage.get().token);

  const loginMutation = useLoginMutation();
  const signupMutation = useSignupMutation();
  const logoutMutation = useLogoutMutation();

  const {
    data: user,
    isLoading: isProfileLoading,
    isError: isProfileError,
    isSuccess: isProfileSuccess,
    refetch: refetchProfile, // Add refetch function
  } = useProfileQuery(localToken); // Pass localToken to enable/disable query

  // --- Effect to manage localToken state from authStorage changes ---
  useEffect(() => {
    const handleStorageChange = () => {
      const currentToken = authStorage.get().token;
      if (currentToken !== localToken) {
        setLocalToken(currentToken);
        if (!currentToken) {
          // If token is cleared from storage, also clear profile data in cache
          queryClient.setQueryData(queryKeys.auth.profile(), null);
        }
      }
    };
    window.addEventListener("storage", handleStorageChange);
    return () => window.removeEventListener("storage", handleStorageChange);
  }, [localToken, queryClient]);

  // --- Effect to connect/disconnect socket based on localToken ---
  useEffect(() => {
    if (localToken) {
      socketService.connect(localToken);
    } else {
      socketService.disconnect();
    }
    // Cleanup on unmount or token change
    return () => socketService.disconnect();
  }, [localToken]);

  // --- Effect to handle profile query errors (e.g., expired session) ---
  useEffect(() => {
    if (isProfileError && localToken) {
      // Only show error if a token was present
      showToast.error("Your session may have expired. Please log in again.");
      authStorage.clear();
      setLocalToken(null); // Clear local token state
      queryClient.clear(); // Clear all cache data
      navigate("/auth?session_expired=true", { replace: true }); // Redirect to auth page
    }
  }, [isProfileError, localToken, queryClient, navigate]);

  // --- Authentication Actions (Memoized for stability) ---
  const login = useCallback(
    async (email, password) => {
      try {
        const response = await loginMutation.mutateAsync({ email, password });
        const { token: newToken, user: newUser } = response.data;
        authStorage.set({ token: newToken, user: newUser, rememberMe: true });
        setLocalToken(newToken); // Update local token state
        queryClient.setQueryData(queryKeys.auth.profile(), newUser); // Optimistically update cache
        showToast.success("Login successful!");
        navigate("/dashboard"); // Centralize navigation
      } catch (error) {
        // Error handling is already in useLoginMutation
      }
    },
    [loginMutation, queryClient, navigate]
  );

  const signup = useCallback(
    async (signupData) => {
      try {
        await signupMutation.mutateAsync(signupData);
        showToast.success("Account created! Please log in to continue.");
        // No token is returned on signup, so no localToken update here
        navigate("/auth?view=login"); // Redirect to login after signup
      } catch (error) {
        // Error handling is already in useSignupMutation
      }
    },
    [signupMutation, navigate]
  );

  const logout = useCallback(() => {
    logoutMutation.mutate(undefined, {
      onSuccess: () => {
        authStorage.clear();
        setLocalToken(null); // Clear local token state
        queryClient.clear(); // Clear all cache data
        showToast.success("You have been logged out.");
        navigate("/auth", { replace: true }); // Centralize navigation
      },
    });
  }, [logoutMutation, queryClient, navigate]);

  const value = {
    user: isProfileSuccess ? user : null,
    token: localToken, // Expose localToken
    isAuthenticated: !!localToken && isProfileSuccess,
    isLoading: isProfileLoading && !!localToken,
    login,
    signup,
    logout,
    isLoggingIn: loginMutation.isPending,
    isSigningUp: signupMutation.isPending,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
