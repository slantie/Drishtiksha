// src/components/PublicRoute.jsx

import React from "react";
import { Navigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext.jsx";
import { PageLoader } from "./ui/LoadingSpinner.jsx";

const PublicRoute = ({ children }) => {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return <PageLoader text="Loading..." />;
  }

  if (isAuthenticated) {
    // If the user is already logged in, redirect them from public-only pages (like login) to the main dashboard.
    return <Navigate to="/dashboard" replace />;
  }

  return children;
};

export default PublicRoute;
