/**
 * @file src/utils/toast.js
 * @description Toast utility and helpers for notifications
 */

import toast from "react-hot-toast";

// Default toast options
const defaultOptions = {
    duration: 4000,
    position: "bottom-right",
};

// Main toast functions
export const showToast = {
    success: (message, options = {}) =>
        toast.success(message, { ...defaultOptions, ...options }),
    error: (message, options = {}) =>
        toast.error(message, { ...defaultOptions, ...options }),
    warning: (message, options = {}) =>
        toast(message, { icon: "⚠️", ...defaultOptions, ...options }),
    info: (message, options = {}) =>
        toast(message, { icon: "ℹ️", ...defaultOptions, ...options }),
    loading: (message, options = {}) =>
        toast.loading(message, { ...defaultOptions, ...options }),
    promise: (promise, messages, options = {}) =>
        toast.promise(promise, messages, { ...defaultOptions, ...options }),
    dismiss: (toastId) => toast.dismiss(toastId),
    dismissAll: () => toast.dismiss(),
};

// Validation toast helpers
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
    terms: () =>
        showToast.warning("You must agree to the terms and conditions"),
};

// Auth toast helpers
export const authToast = {
    loginSuccess: () => showToast.success("Logged In!", { duration: 2000 }),
    loginError: (message) =>
        showToast.error(message || "Login failed. Please try again."),
    signupSuccess: () =>
        showToast.success("Account created successfully! Welcome aboard!", {
            duration: 3000,
        }),
    signupError: (message) =>
        showToast.error(message || "Failed to create account"),
    authenticating: () => showToast.loading("Authenticating..."),
    creatingAccount: () => showToast.loading("Creating your account..."),
    logoutSuccess: () =>
        showToast.success("Logged out successfully!", { duration: 2000 }),
    sessionExpired: () =>
        showToast.warning("Your session has expired. Please login again.", {
            duration: 5000,
        }),
    connectionError: () =>
        showToast.error(
            "Connection error. Please check your internet and try again.",
            { duration: 6000 }
        ),
    passwordResetSent: () =>
        showToast.success("Password reset email sent! Check your inbox.", {
            duration: 5000,
        }),
    passwordResetError: () =>
        showToast.error(
            "Failed to send password reset email. Please try again."
        ),
    emailVerificationSent: () =>
        showToast.success("Verification email sent! Please check your inbox.", {
            duration: 5000,
        }),
};

// App action toast helpers
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
        showToast.success(`${item} copied to clipboard!`, { duration: 2000 }),
    copyError: () => showToast.error("Failed to copy to clipboard"),
    networkError: () =>
        showToast.error("Network error. Please check your connection.", {
            duration: 5000,
        }),
    permissionDenied: () =>
        showToast.error("You don't have permission to perform this action"),
    comingSoon: () =>
        showToast.info("This feature is coming soon!", { duration: 3000 }),
};

export { toast };
export default showToast;
