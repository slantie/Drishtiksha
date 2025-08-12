// src/hooks/useProfile.js

import { useState, useEffect, useCallback } from "react";
import { showToast } from "../utils/toast";
import { API_ENDPOINTS } from "../constants/apiEndpoints";

export const useProfile = () => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const [editMode, setEditMode] = useState(false);
    const [passwordMode, setPasswordMode] = useState(false);
    const [avatarMode, setAvatarMode] = useState(false);

    // Form states
    const [profileForm, setProfileForm] = useState({
        firstName: "",
        lastName: "",
        bio: "",
        phone: "",
    });

    const [passwordForm, setPasswordForm] = useState({
        currentPassword: "",
        newPassword: "",
        confirmPassword: "",
    });

    const [avatarForm, setAvatarForm] = useState({
        avatar: "",
    });

    const [showPasswords, setShowPasswords] = useState({
        current: false,
        new: false,
        confirm: false,
    });

    // Get auth token
    const getAuthToken = () => {
        return (
            localStorage.getItem("authToken") ||
            sessionStorage.getItem("authToken")
        );
    };

    // Fetch user profile
    const fetchProfile = useCallback(async () => {
        try {
            const token = getAuthToken();
            if (!token) {
                showToast.error("Please log in to view your profile");
                window.location.href = "/auth?view=login";
                return;
            }

            const response = await fetch(API_ENDPOINTS.GET_PROFILE, {
                method: "GET",
                headers: {
                    Authorization: `Bearer ${token}`,
                    "Content-Type": "application/json",
                },
                credentials: "include",
            });

            const data = await response.json();

            if (response.ok && data.success) {
                setUser(data.data);
                setProfileForm({
                    firstName: data.data.firstName || "",
                    lastName: data.data.lastName || "",
                    bio: data.data.bio || "",
                    phone: data.data.phone || "",
                });
                setAvatarForm({
                    avatar: data.data.avatar || "",
                });
            } else {
                showToast.error(data.message || "Failed to load profile");
            }
        } catch (error) {
            console.error("Profile fetch error:", error);
            showToast.error("Failed to load profile");
        } finally {
            setLoading(false);
        }
    }, []);

    // Update profile
    const updateProfile = async () => {
        try {
            // Validate form data before sending
            if (
                profileForm.firstName &&
                profileForm.firstName.trim().length < 2
            ) {
                showToast.error(
                    "First name must be at least 2 characters long"
                );
                return;
            }

            if (
                profileForm.lastName &&
                profileForm.lastName.trim().length < 2
            ) {
                showToast.error("Last name must be at least 2 characters long");
                return;
            }

            if (profileForm.bio && profileForm.bio.length > 500) {
                showToast.error("Bio must not exceed 500 characters");
                return;
            }

            // Clean and validate phone number
            let cleanPhone = null;
            if (profileForm.phone && profileForm.phone.trim()) {
                cleanPhone = profileForm.phone.trim();
                // Validate Indian phone number format: +91 followed by 10 digits starting with 6-9
                const phoneRegex = /^\+91\s?[6-9]\d{9}$/;
                if (!phoneRegex.test(cleanPhone)) {
                    showToast.error(
                        "Please enter a valid Indian mobile number (format: +91 9876543210)"
                    );
                    return;
                }
                // Remove any spaces for consistent storage
                cleanPhone = cleanPhone.replace(/\s/g, "");
            }

            const token = getAuthToken();
            const loadingToast = showToast.loading("Updating profile...");

            // Prepare clean data
            const updateData = {
                firstName: profileForm.firstName?.trim() || undefined,
                lastName: profileForm.lastName?.trim() || undefined,
                bio: profileForm.bio?.trim() || undefined,
                phone: cleanPhone || null,
            };

            // Remove undefined values
            Object.keys(updateData).forEach((key) => {
                if (updateData[key] === undefined) {
                    delete updateData[key];
                }
            });

            const response = await fetch(API_ENDPOINTS.UPDATE_PROFILE, {
                method: "PUT",
                headers: {
                    Authorization: `Bearer ${token}`,
                    "Content-Type": "application/json",
                },
                credentials: "include",
                body: JSON.stringify(updateData),
            });

            const data = await response.json();
            showToast.dismiss(loadingToast);

            if (response.ok && data.success) {
                setUser(data.data);
                setEditMode(false);
                showToast.success("Profile updated successfully!");

                // Update stored user data
                const storage = localStorage.getItem("authToken")
                    ? localStorage
                    : sessionStorage;
                storage.setItem("user", JSON.stringify(data.data));
            } else {
                // Handle validation errors from backend
                if (data.errors && Array.isArray(data.errors)) {
                    const errorMessages = data.errors
                        .map((err) => err.message)
                        .join(", ");
                    showToast.error(`Validation errors: ${errorMessages}`);
                } else {
                    showToast.error(data.message || "Failed to update profile");
                }
            }
        } catch (error) {
            console.error("Profile update error:", error);
            showToast.error("Failed to update profile");
        }
    };

    // Update password
    const updatePassword = async () => {
        try {
            if (passwordForm.newPassword !== passwordForm.confirmPassword) {
                showToast.error("New passwords do not match");
                return;
            }

            const token = getAuthToken();
            const loadingToast = showToast.loading("Updating password...");

            const response = await fetch(API_ENDPOINTS.UPDATE_PASSWORD, {
                method: "PUT",
                headers: {
                    Authorization: `Bearer ${token}`,
                    "Content-Type": "application/json",
                },
                credentials: "include",
                body: JSON.stringify(passwordForm),
            });

            const data = await response.json();
            showToast.dismiss(loadingToast);

            if (response.ok && data.success) {
                setPasswordMode(false);
                setPasswordForm({
                    currentPassword: "",
                    newPassword: "",
                    confirmPassword: "",
                });
                showToast.success("Password updated successfully!");
            } else {
                showToast.error(data.message || "Failed to update password");
            }
        } catch (error) {
            console.error("Password update error:", error);
            showToast.error("Failed to update password");
        }
    };

    // Update avatar
    const updateAvatar = async () => {
        try {
            const token = getAuthToken();
            const loadingToast = showToast.loading("Updating avatar...");

            const response = await fetch(API_ENDPOINTS.UPDATE_AVATAR, {
                method: "PUT",
                headers: {
                    Authorization: `Bearer ${token}`,
                    "Content-Type": "application/json",
                },
                credentials: "include",
                body: JSON.stringify(avatarForm),
            });

            const data = await response.json();
            showToast.dismiss(loadingToast);

            if (response.ok && data.success) {
                setUser(data.data);
                setAvatarMode(false);
                showToast.success("Avatar updated successfully!");

                // Update stored user data
                const storage = localStorage.getItem("authToken")
                    ? localStorage
                    : sessionStorage;
                storage.setItem("user", JSON.stringify(data.data));
            } else {
                showToast.error(data.message || "Failed to update avatar");
            }
        } catch (error) {
            console.error("Avatar update error:", error);
            showToast.error("Failed to update avatar");
        }
    };

    // Delete avatar
    const deleteAvatar = async () => {
        try {
            const token = getAuthToken();
            const loadingToast = showToast.loading("Removing avatar...");

            const response = await fetch(API_ENDPOINTS.DELETE_AVATAR, {
                method: "DELETE",
                headers: {
                    Authorization: `Bearer ${token}`,
                    "Content-Type": "application/json",
                },
                credentials: "include",
            });

            const data = await response.json();
            showToast.dismiss(loadingToast);

            if (response.ok && data.success) {
                setUser(data.data);
                setAvatarForm({ avatar: "" });
                showToast.success("Avatar removed successfully!");

                // Update stored user data
                const storage = localStorage.getItem("authToken")
                    ? localStorage
                    : sessionStorage;
                storage.setItem("user", JSON.stringify(data.data));
            } else {
                showToast.error(data.message || "Failed to remove avatar");
            }
        } catch (error) {
            console.error("Avatar delete error:", error);
            showToast.error("Failed to remove avatar");
        }
    };

    // Handle form changes
    const handleProfileChange = (e) => {
        const { name, value } = e.target;

        // Validate phone number format for Indian mobile numbers
        if (name === "phone" && value.length > 0) {
            const phoneRegex = /^\+91\s?[6-9]\d{9}$/;
            if (!phoneRegex.test(value)) {
                // Show validation hint but still allow typing
                console.log(
                    "Phone format should be +91 followed by 10-digit mobile number starting with 6-9"
                );
            }
        }

        setProfileForm({
            ...profileForm,
            [name]: value,
        });
    };

    const handlePasswordChange = (e) => {
        setPasswordForm({
            ...passwordForm,
            [e.target.name]: e.target.value,
        });
    };

    const handleAvatarChange = (e) => {
        setAvatarForm({
            ...avatarForm,
            [e.target.name]: e.target.value,
        });
    };

    // Toggle password visibility
    const togglePasswordVisibility = (field) => {
        setShowPasswords({
            ...showPasswords,
            [field]: !showPasswords[field],
        });
    };

    // Cancel edit modes
    const cancelEdit = () => {
        setEditMode(false);
        setProfileForm({
            firstName: user?.firstName || "",
            lastName: user?.lastName || "",
            bio: user?.bio || "",
            phone: user?.phone || "",
        });
    };

    const cancelPassword = () => {
        setPasswordMode(false);
        setPasswordForm({
            currentPassword: "",
            newPassword: "",
            confirmPassword: "",
        });
    };

    const cancelAvatar = () => {
        setAvatarMode(false);
        setAvatarForm({
            avatar: user?.avatar || "",
        });
    };

    // Initialize profile data on mount
    useEffect(() => {
        fetchProfile();
    }, [fetchProfile]);

    return {
        // State
        user,
        loading,
        editMode,
        passwordMode,
        avatarMode,
        profileForm,
        passwordForm,
        avatarForm,
        showPasswords,

        // Actions
        setEditMode,
        setPasswordMode,
        setAvatarMode,
        updateProfile,
        updatePassword,
        updateAvatar,
        deleteAvatar,
        handleProfileChange,
        handlePasswordChange,
        handleAvatarChange,
        togglePasswordVisibility,
        cancelEdit,
        cancelPassword,
        cancelAvatar,
        fetchProfile,
    };
};
