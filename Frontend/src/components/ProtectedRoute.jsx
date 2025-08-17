// src/components/ProtectedRoute.jsx

import React from "react";
import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext.jsx";
import { PageLoader } from "./ui/LoadingSpinner.jsx"; // REFACTOR: Using our standardized PageLoader.

const ProtectedRoute = ({ children, roles = [] }) => {
    const { isAuthenticated, user, isLoading } = useAuth();
    const location = useLocation();

    if (isLoading) {
        // REFACTOR: Provides a full-screen, consistent loading experience.
        return <PageLoader text="Verifying session..." />;
    }

    if (!isAuthenticated) {
        return (
            <Navigate
                to={`/auth?redirect=${encodeURIComponent(location.pathname)}`}
                replace
            />
        );
    }

    if (roles.length > 0 && !roles.includes(user?.role)) {
        return <Navigate to="/dashboard" replace />;
    }

    return children;
};

export default ProtectedRoute;
