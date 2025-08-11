import React from "react";
import { Navigate } from "react-router-dom";

const PublicRoute = ({ children, redirectPath = "/" }) => {
  // Check if user is already authenticated
  const getAuthData = () => {
    try {
      const token =
        localStorage.getItem("authToken") ||
        sessionStorage.getItem("authToken");
      const authStatus =
        localStorage.getItem("isAuthenticated") ||
        sessionStorage.getItem("isAuthenticated");
      return token && authStatus === "true";
    } catch (error) {
      console.error("Error checking auth status:", error);
      return false;
    }
  };

  const isAuthenticated = getAuthData();

  // If user is authenticated, redirect them away from login/signup pages
  if (isAuthenticated) {
    return <Navigate to={redirectPath} replace />;
  }

  // If not authenticated, show the public page (login/signup)
  return children;
};

export default PublicRoute;
