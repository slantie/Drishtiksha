// src/components/PublicRoute.jsx

import React from "react";
import { Navigate } from "react-router-dom";
import { useAuth } from "../hooks/useAuth.js";
import LoadingSpinner from "./ui/LoadingSpinner.jsx";

const PublicRoute = ({ children }) => {
    const { isAuthenticated, isLoading } = useAuth();

    if (isLoading) {
        return (
            <div className="flex h-screen items-center justify-center">
                <LoadingSpinner />
            </div>
        );
    }

    if (isAuthenticated) {
        // If the user is logged in, redirect them away from public pages (like login/signup)
        return <Navigate to="/dashboard" replace />;
    }

    // If not authenticated, render the public page.
    return children;
};

export default PublicRoute;
