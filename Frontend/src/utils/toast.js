/**
 * @file src/utils/toast.js
 * @description Toast utility and helpers for notifications
 */

import toast from "react-hot-toast";
import React from "react"; // Import React to use JSX in toast messages
import { ToastProgress } from "../components/ui/ToastProgress.jsx"; // Import the updated ToastProgress component

// Default toast options
const defaultOptions = {
  duration: 4000,
  position: "bottom-right",
  style: {
    borderRadius: "10px",
    backgroundColor: "var(--toast-bg)", // Use CSS variable from index.css
    color: "var(--toast-text)", // Use CSS variable from index.css
    border: "1px solid var(--toast-border)", // Use CSS variable from index.css
    boxShadow: "var(--toast-shadow)", // Use CSS variable from index.css
  },
};

// Main toast functions
export const showToast = {
  success: (message, options = {}) =>
    toast.success(<ToastProgress message={message} />, {
      ...defaultOptions,
      ...options,
      icon: "✅",
    }), // Use ToastProgress
  error: (message, options = {}) =>
    toast.error(<ToastProgress message={message} isError={true} />, {
      ...defaultOptions,
      ...options,
      icon: "❌",
    }), // Use ToastProgress, pass isError
  warning: (message, options = {}) =>
    toast(<ToastProgress message={message} />, {
      ...defaultOptions,
      ...options,
      icon: "⚠️",
    }), // Use ToastProgress
  info: (message, options = {}) =>
    toast(<ToastProgress message={message} />, {
      ...defaultOptions,
      ...options,
      icon: "ℹ️",
    }), // Use ToastProgress
  loading: (message, options = {}) =>
    toast.loading(<ToastProgress message={message} />, {
      ...defaultOptions,
      ...options,
      duration: Infinity,
      icon: null,
    }), // Use ToastProgress, icon handled internally

  promise: (promise, messages, options = {}) =>
    toast.promise(
      promise,
      {
        loading: <ToastProgress message={messages.loading} />,
        success: (data) => <ToastProgress message={messages.success(data)} />,
        error: (error) => (
          <ToastProgress message={messages.error(error)} isError={true} />
        ),
      },
      { ...defaultOptions, ...options, icon: null } // Icons handled by ToastProgress
    ),
  dismiss: (toastId) => toast.dismiss(toastId),
  dismissAll: () => toast.dismiss(),
};

// Validation toast helpers (no change needed in logic, just consistent usage of showToast)
export const validationToast = {
  required: (field) => showToast.warning(`${field} is required`),
  invalid: (field) => showToast.warning(`Please enter a valid ${field}`),
  mismatch: (field1, field2) =>
    showToast.error(`${field1} and ${field2} do not match`),
  minLength: (field, length) =>
    showToast.warning(`${field} must be at least ${length} characters`),
  maxLength: (field, length) =>
    showToast.warning(`${field} cannot exceed ${length} characters`),
  email: () => showToast.warning("Please enter a valid email address"),
  phone: () => showToast.warning("Please enter a valid phone number"),
  terms: () => showToast.warning("You must agree to the terms and conditions"),
};

// Auth toast helpers (no change needed in logic, just consistent usage of showToast)
export const authToast = {
  loginSuccess: () => showToast.success("Logged In!"),
  loginError: (message) =>
    showToast.error(message || "Login failed. Please try again."),
  signupSuccess: () =>
    showToast.success("Account created successfully! Welcome aboard!"),
  signupError: (message) =>
    showToast.error(message || "Failed to create account"),
  authenticating: () => showToast.loading("Authenticating..."),
  creatingAccount: () => showToast.loading("Creating your account..."),
  logoutSuccess: () => showToast.success("Logged out successfully!"),
  sessionExpired: () =>
    showToast.warning("Your session has expired. Please login again."),
  connectionError: () =>
    showToast.error(
      "Connection error. Please check your internet and try again."
    ),
  passwordResetSent: () =>
    showToast.success("Password reset email sent! Check your inbox."),
  passwordResetError: () =>
    showToast.error("Failed to send password reset email. Please try again."),
  emailVerificationSent: () =>
    showToast.success("Verification email sent! Please check your inbox."),
};

// App action toast helpers (no change needed in logic, just consistent usage of showToast)
export const appToast = {
  saveSuccess: (item = "Data") =>
    showToast.success(`${item} saved successfully!`),
  saveError: (item = "Data") =>
    showToast.error(`Failed to save ${item.toLowerCase()}`),
  deleteSuccess: (item = "Item") =>
    showToast.success(`${item} deleted successfully!`),
  deleteError: (item = "Item") =>
    showToast.error(`Failed to delete ${item.toLowerCase()}`),
  updateSuccess: (item = "Data") =>
    showToast.success(`${item} updated successfully!`),
  updateError: (item = "Data") =>
    showToast.error(`Failed to update ${item.toLowerCase()}`),
  uploadSuccess: (item = "File") =>
    showToast.success(`${item} uploaded successfully!`),
  uploadError: (item = "File") =>
    showToast.error(`Failed to upload ${item.toLowerCase()}`),
  copySuccess: (item = "Content") =>
    showToast.success(`${item} copied to clipboard!`),
  copyError: () => showToast.error("Failed to copy to clipboard"),
  networkError: () =>
    showToast.error("Network error. Please check your connection."),
  permissionDenied: () =>
    showToast.error("You don't have permission to perform this action"),
  comingSoon: () => showToast.info("This feature is coming soon!"),
};

export { toast };
export default showToast;
