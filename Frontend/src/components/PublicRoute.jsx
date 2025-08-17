// src/components/PublicRoute.jsx

import React from "react";
import { Navigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext.jsx";
import { PageLoader } from "./ui/LoadingSpinner.jsx"; // REFACTOR: Using our standardized PageLoader.

const PublicRoute = ({ children }) => {
    const { isAuthenticated, isLoading } = useAuth();

    if (isLoading) {
        // REFACTOR: Ensures a consistent loading UI for public routes as well.
        return <PageLoader text="Loading..." />;
    }

    if (isAuthenticated) {
        return <Navigate to="/dashboard" replace />;
    }

    return children;
};

export default PublicRoute;
