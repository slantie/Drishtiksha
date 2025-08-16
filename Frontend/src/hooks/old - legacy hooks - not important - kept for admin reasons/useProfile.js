// src/hooks/useProfile.js
// ⚠️ DEPRECATED: This hook is deprecated. Use hooks from useAuthQuery.js instead.

import { useState } from "react";
import {
    useProfileQuery,
    useUpdateProfileMutation,
    useUpdatePasswordMutation,
    useUpdateAvatarMutation,
    useDeleteAvatarMutation,
} from "./useAuthQuery.js";

/**
 * @deprecated Use individual hooks from useAuthQuery.js instead:
 * - useProfileQuery() for fetching profile
 * - useUpdateProfileMutation() for updating profile
 * - useUpdatePasswordMutation() for updating password
 * - useUpdateAvatarMutation() / useDeleteAvatarMutation() for avatar management
 *
 * This legacy wrapper is kept for compatibility but will be removed.
 */
export const useProfile = () => {
    console.warn(
        "⚠️ useProfile hook is deprecated. Please use individual hooks from useAuthQuery.js instead. " +
            "See TANSTACK_MIGRATION.md for migration guide."
    );
    // TanStack Query hooks
    const {
        data: user,
        isLoading: loading,
        refetch: fetchProfile,
    } = useProfileQuery();
    const updateProfileMutation = useUpdateProfileMutation();
    const updatePasswordMutation = useUpdatePasswordMutation();
    const updateAvatarMutation = useUpdateAvatarMutation();
    const deleteAvatarMutation = useDeleteAvatarMutation();

    // UI state
    const [editMode, setEditMode] = useState(false);
    const [passwordMode, setPasswordMode] = useState(false);
    const [avatarMode, setAvatarMode] = useState(false);

    // Form states
    const [profileForm, setProfileForm] = useState({
        firstName: user?.firstName || "",
        lastName: user?.lastName || "",
        bio: user?.bio || "",
        phone: user?.phone || "",
    });

    const [passwordForm, setPasswordForm] = useState({
        currentPassword: "",
        newPassword: "",
        confirmPassword: "",
    });

    const [avatarForm, setAvatarForm] = useState({
        avatar: user?.avatar || "",
    });

    const [showPasswords, setShowPasswords] = useState({
        current: false,
        new: false,
        confirm: false,
    });

    // Update profile
    const updateProfile = async () => {
        // Validation logic (same as before)
        if (profileForm.firstName && profileForm.firstName.trim().length < 2) {
            return; // Let TanStack mutation handle errors
        }

        const updateData = {
            firstName: profileForm.firstName?.trim() || undefined,
            lastName: profileForm.lastName?.trim() || undefined,
            bio: profileForm.bio?.trim() || undefined,
            phone: profileForm.phone?.replace(/\s/g, "") || null,
        };

        // Remove undefined values
        Object.keys(updateData).forEach((key) => {
            if (updateData[key] === undefined) {
                delete updateData[key];
            }
        });

        await updateProfileMutation.mutateAsync(updateData);
        setEditMode(false);
    };

    // Update password
    const updatePassword = async () => {
        if (passwordForm.newPassword !== passwordForm.confirmPassword) {
            return; // Let user handle validation
        }

        await updatePasswordMutation.mutateAsync(passwordForm);
        setPasswordMode(false);
        setPasswordForm({
            currentPassword: "",
            newPassword: "",
            confirmPassword: "",
        });
    };

    // Update avatar
    const updateAvatar = async () => {
        const formData = new FormData();
        formData.append("avatar", avatarForm.avatar);

        await updateAvatarMutation.mutateAsync(formData);
        setAvatarMode(false);
    };

    // Delete avatar
    const deleteAvatar = async () => {
        await deleteAvatarMutation.mutateAsync();
        setAvatarForm({ avatar: "" });
    };

    // Handle form changes
    const handleProfileChange = (e) => {
        const { name, value } = e.target;
        setProfileForm({ ...profileForm, [name]: value });
    };

    const handlePasswordChange = (e) => {
        setPasswordForm({ ...passwordForm, [e.target.name]: e.target.value });
    };

    const handleAvatarChange = (e) => {
        setAvatarForm({ ...avatarForm, [e.target.name]: e.target.value });
    };

    // Toggle password visibility
    const togglePasswordVisibility = (field) => {
        setShowPasswords({ ...showPasswords, [field]: !showPasswords[field] });
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
        setAvatarForm({ avatar: user?.avatar || "" });
    };

    return {
        // State
        user,
        loading:
            loading ||
            updateProfileMutation.isPending ||
            updatePasswordMutation.isPending ||
            updateAvatarMutation.isPending ||
            deleteAvatarMutation.isPending,
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
