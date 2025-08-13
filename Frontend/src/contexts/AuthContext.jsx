// src/contexts/AuthContext.jsx

import React, { createContext, useState, useEffect } from "react";
import { authApiService } from "../services/apiService.js";
import { useNavigate } from "react-router-dom";

export const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [token, setToken] = useState(
        sessionStorage.getItem("authToken") || localStorage.getItem("authToken")
    );
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [isLoading, setIsLoading] = useState(true);
    const navigate = useNavigate();

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

    const login = async (email, password, rememberMe) => {
        const data = await authApiService.login(email, password);
        const storage = rememberMe ? localStorage : sessionStorage;

        storage.setItem("authToken", data.data.token);
        storage.setItem("user", JSON.stringify(data.data.user));

        setToken(data.data.token);
        setUser(data.data.user);
        setIsAuthenticated(true);

        console.log("User Data:", data.data.user)

        // Only navigate to dashboard after successful login
        navigate("/dashboard");
    };

    const signup = async (signupData) => {
        await authApiService.signup(signupData);
    };

    const logout = () => {
        sessionStorage.removeItem("authToken");
        sessionStorage.removeItem("user");
        localStorage.removeItem("authToken");
        localStorage.removeItem("user");

        setToken(null);
        setUser(null);
        setIsAuthenticated(false);

        navigate("/auth");
    };

    const value = {
        user,
        token,
        isAuthenticated,
        isLoading,
        login,
        signup,
        logout,
    };

    return (
        <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
    );
};
