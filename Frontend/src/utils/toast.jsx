/**
 * @file src/utils/toast.jsx
 * @description Toast utility for notifications, now using default icons for simplicity.
 */

import toast from "react-hot-toast";
import React from "react";
import { ToastProgress } from "../components/ui/ToastProgress.jsx";

// Default toast options are now managed in ToastProvider for consistency.
const defaultOptions = {
  duration: 4000,
};

export const showToast = {
  // Pass a simple string to use the default animated icon from react-hot-toast.
  success: (message, options = {}) =>
    toast.success(message, {
      ...defaultOptions,
      ...options,
    }),

  // Pass a simple string for error messages as well.
  error: (message, options = {}) =>
    toast.error(message, {
      ...defaultOptions,
      ...options,
    }),

  // For loading and progress, we still use our custom component.
  loading: (content, options = {}) => {
    // If content is a string, wrap it in ToastProgress for consistency.
    const messageContent =
      typeof content === "string" ? (
        <ToastProgress message={content} />
      ) : (
        content
      );
    return toast.loading(messageContent, {
      ...defaultOptions,
      duration: Infinity, // Loading toasts should persist until dismissed.
      ...options,
    });
  },

  // A generic toast for info/warning, which can use a custom component if needed.
  info: (message, options = {}) =>
    toast(message, {
      // 'toast()' is the default, non-iconed version
      ...defaultOptions,
      ...options,
      icon: "ℹ️",
    }),

  warning: (message, options = {}) =>
    toast(message, {
      ...defaultOptions,
      ...options,
      icon: "⚠️",
    }),

  // Dismiss functions remain the same.
  dismiss: (toastId) => toast.dismiss(toastId),
  dismissAll: () => toast.dismiss(),
};

export default showToast;
