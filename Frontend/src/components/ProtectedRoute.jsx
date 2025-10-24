// src/components/ProtectedRoute.jsx

import React from "react";
import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "../hooks/useAuth.js";
import { PageLoader } from "./ui/LoadingSpinner.jsx";

const ProtectedRoute = ({ children, roles = [] }) => {
  const { isAuthenticated, user, isLoading } = useAuth();
  const location = useLocation();

  if (isLoading) {
    // While the auth state is being determined, show a full-screen loader.
    return <PageLoader text="Verifying session..." />;
  }

  if (!isAuthenticated) {
    // If the user is not authenticated, redirect them to the auth page.
    // We pass the current location so they can be redirected back after logging in.
    return (
      <Navigate
        to={`/auth?redirect=${encodeURIComponent(location.pathname)}`}
        replace
      />
    );
  }

  // If the route requires specific roles and the user doesn't have one, redirect.
  // Note: user.role is expected to be defined if isAuthenticated is true.
  if (roles.length > 0 && user && !roles.includes(user.role)) {
    return <Navigate to="/dashboard" replace />;
  }

  return children;
};

export default ProtectedRoute;
