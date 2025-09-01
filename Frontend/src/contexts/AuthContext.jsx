// src/contexts/AuthContext.jsx

import React, { createContext, useContext, useEffect, useState } from "react";
import {
  useLoginMutation,
  useSignupMutation,
  useLogoutMutation,
  useProfileQuery,
} from "../hooks/useAuthQuery.js";
import { socketService } from "../lib/socket.jsx";
import { showToast } from "../utils/toast.js";
import { authStorage } from "../utils/authStorage.js";
import { useQueryClient } from "@tanstack/react-query";

export const AuthContext = createContext(null);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used within an AuthProvider");
  return context;
};

export const AuthProvider = ({ children }) => {
  const [token, setToken] = useState(authStorage.get().token);
  const queryClient = useQueryClient();

  const loginMutation = useLoginMutation();
  const signupMutation = useSignupMutation();
  const logoutMutation = useLogoutMutation();

  const {
    data: user,
    isLoading: isProfileLoading,
    isError: isProfileError,
    isSuccess: isProfileSuccess,
  } = useProfileQuery();

  useEffect(() => {
    const handleStorageChange = () => {
      const currentToken = authStorage.get().token;
      setToken(currentToken);
    };
    window.addEventListener("storage", handleStorageChange);
    return () => window.removeEventListener("storage", handleStorageChange);
  }, []);

  useEffect(() => {
    if (token) {
      socketService.connect(token);
    } else {
      socketService.disconnect();
    }
    return () => socketService.disconnect();
  }, [token]);

  useEffect(() => {
    if (isProfileError) {
      showToast.error("Your session may have expired. Please log in again.");
      authStorage.clear();
      setToken(null);
      queryClient.clear();
    }
  }, [isProfileError, queryClient]);

  const login = (email, password) =>
    loginMutation.mutateAsync({ email, password });
  const signup = (signupData) => signupMutation.mutateAsync(signupData);
  const logout = () => logoutMutation.mutate();

  const value = {
    user: isProfileSuccess ? user : null,
    token,
    isAuthenticated: !!token && isProfileSuccess,
    isLoading: isProfileLoading && !!token,
    login,
    signup,
    logout,
    isLoggingIn: loginMutation.isPending,
    isSigningUp: signupMutation.isPending,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
