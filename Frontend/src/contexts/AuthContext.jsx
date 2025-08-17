// src/contexts/AuthContext.jsx

import React, { createContext, useContext, useState, useEffect } from "react";
import {
    useLoginMutation,
    useSignupMutation,
    useLogoutMutation,
    useProfileQuery,
} from "../hooks/useAuthQuery.js";
import { socketService } from "../lib/socket.jsx"; // CORRECTED IMPORT
import { showToast } from "../utils/toast.js";
import { authStorage } from "../utils/authStorage.js";

export const AuthContext = createContext(null);

export const useAuth = () => {
    const context = useContext(AuthContext);

    if (!context) {
        throw new Error("useAuth must be used within an AuthProvider");
    }
    return context;
};

export const useUser = () => {
    const { user } = useAuth();
    return user;
};

export const AuthProvider = ({ children }) => {
    const [token, setToken] = useState(() => {
        return sessionStorage.getItem("authToken");
    });

    const loginMutation = useLoginMutation();
    const signupMutation = useSignupMutation();
    const logoutMutation = useLogoutMutation();

    const {
        data: user,
        isLoading: isProfileLoading,
        isError: isProfileError,
        isSuccess: isProfileSuccess,
    } = useProfileQuery(!!token);

    useEffect(() => {
        if (token && isProfileSuccess) {
            socketService.connect(token);
        } else {
            socketService.disconnect();
        }
        return () => {
            socketService.disconnect();
        };
    }, [token, isProfileSuccess]);

    useEffect(() => {
        if (isProfileError) {
            showToast.error("Your session has expired. Please log in again.");
            setToken(null);
            sessionStorage.removeItem("authToken");
            sessionStorage.removeItem("user");
        }
    }, [isProfileError]);

    const login = (email, password) => {
        return loginMutation
            .mutateAsync({ email, password })
            .then((response) => {
                const authToken = response.data.token;
                setToken(authToken);
            });
    };

    const signup = (signupData) => {
        return signupMutation.mutateAsync(signupData);
    };

    const logout = () => {
        logoutMutation.mutate(undefined, {
            onSuccess: () => {
                setToken(null);
            },
        });
    };

    const value = {
        user,
        token,
        isAuthenticated: !!token && isProfileSuccess,
        isLoading: isProfileLoading && !!token,
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
