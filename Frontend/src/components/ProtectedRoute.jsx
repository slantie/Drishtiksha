import React from "react";
import { showToast } from "../utils/toast";
import { Navigate, useLocation } from "react-router-dom";

const ProtectedRoute = ({
  children,
  roles = [],
  fallbackPath = "/auth?view=login",
  requireAuth = true,
}) => {
  const location = useLocation();

  // Check authentication status
  const getAuthData = () => {
    try {
      const token =
        localStorage.getItem("authToken") ||
        sessionStorage.getItem("authToken");
      const userData =
        localStorage.getItem("user") || sessionStorage.getItem("user");
      const authStatus =
        localStorage.getItem("isAuthenticated") ||
        sessionStorage.getItem("isAuthenticated");

      if (token && userData && authStatus === "true") {
        return {
          isAuthenticated: true,
          user: JSON.parse(userData),
          token,
        };
      }
      return { isAuthenticated: false, user: null, token: null };
    } catch (error) {
      console.error("Error checking auth status:", error);
      return { isAuthenticated: false, user: null, token: null };
    }
  };

  const { isAuthenticated, user } = getAuthData();

  // If authentication is required but user is not authenticated
  if (requireAuth && !isAuthenticated) {
    showToast.error("Please log in to access this page");
    // Redirect to login with current location as return URL
    return (
      <Navigate
        to={`${fallbackPath}&redirect=${encodeURIComponent(location.pathname)}`}
        replace
      />
    );
  }

  // If specific roles are required, check user role
  if (isAuthenticated && roles.length > 0 && user) {
    const userRole = user.role?.toUpperCase();
    const hasRequiredRole = roles.some(
      (role) => role.toUpperCase() === userRole
    );

    if (!hasRequiredRole) {
      showToast.error("You don't have permission to access this page");
      return <Navigate to="/" replace />;
    }
  }

  // If all checks pass, render the protected content
  return children;
};

export default ProtectedRoute;
