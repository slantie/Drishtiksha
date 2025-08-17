// src/components/ProtectedRoute.jsx

import React from "react";
import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext.jsx"; // Updated import path
import LoadingSpinner from "./ui/LoadingSpinner.jsx";

const ProtectedRoute = ({ children, roles = [] }) => {
    const { isAuthenticated, user, isLoading } = useAuth();
    const location = useLocation();

    if (isLoading) {
        return (
            <div className="flex h-screen items-center justify-center">
                <LoadingSpinner />
            </div>
        );
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
