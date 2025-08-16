// src/contexts/AuthContext.jsx

import React, { createContext, useState, useEffect } from "react";
import {
    useLoginMutation,
    useSignupMutation,
    useLogoutMutation,
    useProfileQuery,
} from "../hooks/useAuthQuery.js";
import { socketService } from "../lib/socket.js";

export const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [token, setToken] = useState(
        sessionStorage.getItem("authToken") || localStorage.getItem("authToken")
    );
    const [isAuthenticated, setIsAuthenticated] = useState(!!token);
    const [isLoading, setIsLoading] = useState(true);

    const loginMutation = useLoginMutation();
    const signupMutation = useSignupMutation();
    const logoutMutation = useLogoutMutation();

    const { data: profileData, isError: isProfileError } = useProfileQuery();

    // Effect to initialize auth state from storage
    useEffect(() => {
        const storedToken =
            sessionStorage.getItem("authToken") ||
            localStorage.getItem("authToken");
        if (storedToken) {
            setToken(storedToken);
            setIsAuthenticated(true);
            socketService.connect();
        }
        setIsLoading(false);
    }, []);

    // Effect to sync user data from profile query or clear on error
    useEffect(() => {
        if (profileData) {
            setUser(profileData);
            const storage = localStorage.getItem("authToken")
                ? localStorage
                : sessionStorage;
            storage.setItem("user", JSON.stringify(profileData));
        }
        if (isProfileError) {
            // If fetching profile fails (e.g., token is invalid), log the user out.
            logout();
        }
    }, [profileData, isProfileError]);

    const login = async (email, password, rememberMe) => {
        const result = await loginMutation.mutateAsync({
            email,
            password,
            rememberMe,
        });

        const userData = result.data.user;
        const authToken = result.data.token;

        setToken(authToken);
        setUser(userData);
        setIsAuthenticated(true);
        socketService.connect();
    };

    const signup = (signupData) => {
        return signupMutation.mutateAsync(signupData);
    };

    const logout = () => {
        socketService.disconnect();
        setToken(null);
        setUser(null);
        setIsAuthenticated(false);
        logoutMutation.mutate();
    };

    const value = {
        user,
        token,
        isAuthenticated,
        isLoading:
            isLoading || (isAuthenticated && !profileData && !isProfileError),
        login,
        signup,
        logout,
        isLoggingIn: loginMutation.isPending,
        isSigningUp: signupMutation.isPending,
    };

    return (
        <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
    );
};
