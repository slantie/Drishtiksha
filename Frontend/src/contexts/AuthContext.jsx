// src/contexts/AuthContext.jsx

import React, { createContext, useState, useEffect } from "react";
import {
    useLoginMutation,
    useSignupMutation,
    useLogoutMutation,
    useProfileQuery,
} from "../hooks/useAuthQuery.js";

export const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [token, setToken] = useState(
        sessionStorage.getItem("authToken") || localStorage.getItem("authToken")
    );
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [isLoading, setIsLoading] = useState(true);

    // TanStack Query mutations
    const loginMutation = useLoginMutation();
    const signupMutation = useSignupMutation();
    const logoutMutation = useLogoutMutation();

    // Profile query to sync user data
    const { data: profileData } = useProfileQuery();

    useEffect(() => {
        const storedUser =
            sessionStorage.getItem("user") || localStorage.getItem("user");
        if (token && storedUser) {
            try {
                setUser(JSON.parse(storedUser));
                setIsAuthenticated(true);
            } catch (error) {
                console.error("Failed to parse user data from storage", error);
                sessionStorage.clear();
                localStorage.clear();
            }
        }
        setIsLoading(false);
    }, [token]);

    // Update user state when profile data changes
    useEffect(() => {
        if (profileData) {
            setUser(profileData);
            const storage = localStorage.getItem("user")
                ? localStorage
                : sessionStorage;
            storage.setItem("user", JSON.stringify(profileData));
        }
    }, [profileData]);

    const login = async (email, password, rememberMe) => {
        const result = await loginMutation.mutateAsync({
            email,
            password,
            rememberMe,
        });

        // Update local auth state immediately after successful login
        const userData = result.data.user;
        const authToken = result.data.token;

        setToken(authToken);
        setUser(userData);
        setIsAuthenticated(true);

        return result;
    };

    const signup = async (signupData) => {
        return signupMutation.mutateAsync(signupData);
    };

    const logout = () => {
        // Clear local state immediately
        setToken(null);
        setUser(null);
        setIsAuthenticated(false);

        // Clear storage immediately
        localStorage.removeItem("authToken");
        localStorage.removeItem("user");
        sessionStorage.removeItem("authToken");
        sessionStorage.removeItem("user");

        // Then trigger the mutation (which will handle toast and navigation)
        logoutMutation.mutate();
    };

    const value = {
        user,
        token,
        isAuthenticated,
        isLoading:
            isLoading || loginMutation.isPending || signupMutation.isPending,
        login,
        signup,
        logout,
        // Expose mutation states for better UX
        loginError: loginMutation.error,
        signupError: signupMutation.error,
        isLoggingIn: loginMutation.isPending,
        isSigningUp: signupMutation.isPending,
    };

    return (
        <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
    );
};
