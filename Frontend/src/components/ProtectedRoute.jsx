// src/components/ProtectedRoute.jsx

import React from "react";
import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "../hooks/useAuth.js";
import { showToast } from "../utils/toast.js";
import LoadingSpinner from "./ui/LoadingSpinner.jsx"; // Assuming you have a loading spinner

const ProtectedRoute = ({ children, roles = [] }) => {
    const { isAuthenticated, user, isLoading } = useAuth();
    const location = useLocation();

    // 1. Show a loading state while the auth status is being checked.
    if (isLoading) {
        return (
            <div className="flex h-screen items-center justify-center">
                <LoadingSpinner />
            </div>
        );
    }

    // 2. If not authenticated, redirect to the login page.
    if (!isAuthenticated) {
        // The toast can be triggered on the login page itself if needed,
        // but for now, let's keep it simple.
        return (
            <Navigate
                to={`/auth?redirect=${encodeURIComponent(location.pathname)}`}
                replace
            />
        );
    }

    // 3. If the route requires specific roles, check if the user has one.
    if (roles.length > 0 && !roles.includes(user?.role)) {
        // This side-effect is safe because it happens after the redirect check.
        showToast.error("You don't have permission to access this page.");
        return <Navigate to="/dashboard" replace />;
    }

    // 4. If all checks pass, render the child components.
    return children;
};

export default ProtectedRoute;
