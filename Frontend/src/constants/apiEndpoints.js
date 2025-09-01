// src/constants/apiEndpoints.js

// This file is now simplified to provide only the base URL for the backend API.
// Specific endpoint paths are now co-located with their respective api service files
// (e.g., `src/services/api/media.api.js`) for better code organization.

// The base URL for the backend API, configured via environment variables.
export const API_BASE_URL =
  import.meta.env.VITE_BACKEND_URL || "http://localhost:3000";
