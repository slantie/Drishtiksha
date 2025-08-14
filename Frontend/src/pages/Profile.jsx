import React, { useState, useEffect } from "react";
import {
    User,
    Mail,
    Phone,
    Edit3,
    Save,
    X,
    Camera,
    Lock,
    Trash2,
    Calendar,
    Shield,
    CheckCircle,
    AlertCircle,
} from "lucide-react";
// eslint-disable-next-line no-unused-vars
import { motion, AnimatePresence } from "framer-motion";
import { PageLoader } from "../components/ui/LoadingSpinner";
import {
    useProfileQuery,
    useUpdateProfileMutation,
    useUpdatePasswordMutation,
    useUpdateAvatarMutation,
    useDeleteAvatarMutation,
} from "../hooks/useAuthQuery.js";
import { Card } from "../components/ui/Card.jsx";
import { Button } from "../components/ui/Button.jsx";

const Profile = () => {
    // TanStack Query hooks
    const { data: user, isLoading: loading } = useProfileQuery();
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

    // Update form when user data changes
    useEffect(() => {
        if (user) {
            setProfileForm({
                firstName: user.firstName || "",
                lastName: user.lastName || "",
                bio: user.bio || "",
                phone: user.phone || "",
            });
            setAvatarForm({
                avatar: user.avatar || "",
            });
        }
    }, [user]);

    // Animation variants
    const containerVariants = {
        hidden: { opacity: 0 },
        visible: {
            opacity: 1,
            transition: { duration: 0.6, staggerChildren: 0.1 },
        },
    };

    const cardVariants = {
        hidden: { opacity: 0, y: 20 },
        visible: {
            opacity: 1,
            y: 0,
            transition: { duration: 0.5, ease: "easeOut" },
        },
    };

    const modalVariants = {
        hidden: { opacity: 0, scale: 0.95 },
        visible: {
            opacity: 1,
            scale: 1,
            transition: { duration: 0.2, ease: "easeOut" },
        },
        exit: {
            opacity: 0,
            scale: 0.95,
            transition: { duration: 0.2 },
        },
    };

    const overlayVariants = {
        hidden: { opacity: 0 },
        visible: { opacity: 1 },
        exit: { opacity: 0 },
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

    // Update profile
    const updateProfile = async () => {
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

        try {
            await updateProfileMutation.mutateAsync(updateData);
            setEditMode(false);
        } catch (error) {
            console.error("Profile update failed:", error);
        }
    };

    // Update password
    const updatePassword = async () => {
        if (passwordForm.newPassword !== passwordForm.confirmPassword) {
            return;
        }

        try {
            await updatePasswordMutation.mutateAsync(passwordForm);
            setPasswordMode(false);
            setPasswordForm({
                currentPassword: "",
                newPassword: "",
                confirmPassword: "",
            });
        } catch (error) {
            console.error("Password update failed:", error);
        }
    };

    // Update avatar
    const updateAvatar = async () => {
        const avatarData = {
            avatar: avatarForm.avatar,
        };

        try {
            await updateAvatarMutation.mutateAsync(avatarData);
            setAvatarMode(false);
        } catch (error) {
            console.error("Avatar update failed:", error);
        }
    };

    // Delete avatar
    const deleteAvatar = async () => {
        try {
            await deleteAvatarMutation.mutateAsync();
            setAvatarForm({ avatar: "" });
        } catch (error) {
            console.error("Avatar deletion failed:", error);
        }
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

    const cancelAvatar = () => {
        setAvatarMode(false);
        setAvatarForm({ avatar: user?.avatar || "" });
    };

    const isAnyLoading =
        loading ||
        updateProfileMutation.isPending ||
        updatePasswordMutation.isPending ||
        updateAvatarMutation.isPending ||
        deleteAvatarMutation.isPending;

    if (isAnyLoading && !user) {
        return <PageLoader text="Loading Profile" />;
    }

    if (!user) {
        return (
            <motion.div
                className="w-full mx-auto p-6"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5 }}
            >
                <div className="text-center py-16">
                    <motion.div
                        initial={{ scale: 0.8 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: 0.2, duration: 0.4 }}
                    >
                        <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
                        <h2 className="text-2xl font-bold text-light-text dark:text-dark-text mb-4">
                            Profile Not Found
                        </h2>
                        <p className="text-light-muted-text dark:text-dark-muted-text text-lg">
                            Unable to load your profile. Please try logging in
                            again.
                        </p>
                    </motion.div>
                </div>
            </motion.div>
        );
    }

    return (
        <motion.div
            className="w-full mx-auto space-y-6"
            variants={containerVariants}
            initial="hidden"
            animate="visible"
        >
            {/* Header */}
            <motion.div variants={cardVariants}>
                <Card>
                    <div className="p-2">
                        <div className="flex justify-between items-center">
                            <div>
                                <h1 className="text-3xl font-bold ">
                                    Profile Management
                                </h1>
                                <p className="text-light-muted-text dark:text-dark-muted-text mt-2">
                                    Manage your personal information and account
                                    settings.
                                </p>
                            </div>
                            <div className="flex items-center justify-center space-x-3">
                                {!passwordMode && (
                                    <Button
                                        onClick={() => setPasswordMode(true)}
                                        className="flex items-center justify-center space-x-2 rounded-xl"
                                        variant="outline"
                                        asChild
                                    >
                                        <motion.div
                                            whileHover="hover"
                                            whileTap="tap"
                                            initial={{ x: 20, opacity: 0 }}
                                            animate={{ x: 0, opacity: 1 }}
                                            transition={{
                                                delay: 0.3,
                                                duration: 0.5,
                                            }}
                                        >
                                            <Edit3 className="w-4 h-4" />
                                            <span>Change Password</span>
                                        </motion.div>
                                    </Button>
                                )}
                                {!editMode && (
                                    <Button
                                        onClick={() => setEditMode(true)}
                                        className="flex items-center justify-center space-x-2 rounded-xl"
                                        variant="outline"
                                        asChild
                                    >
                                        <motion.div
                                            whileHover="hover"
                                            whileTap="tap"
                                            initial={{ x: 20, opacity: 0 }}
                                            animate={{ x: 0, opacity: 1 }}
                                            transition={{
                                                delay: 0.3,
                                                duration: 0.5,
                                            }}
                                        >
                                            <Edit3 className="w-4 h-4" />
                                            <span>Edit Profile</span>
                                        </motion.div>
                                    </Button>
                                )}
                            </div>
                        </div>
                    </div>
                </Card>
            </motion.div>

            {/* Main Layout Grid */}
            <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
                {/* Left Column - Avatar & Quick Info */}
                <motion.div
                    className="xl:col-span-1 space-y-6"
                    variants={cardVariants}
                >
                    {/* Avatar Card */}
                    <motion.div
                        className="bg-light-background dark:bg-dark-noisy-background border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-2xl shadow-xl overflow-hidden"
                        whileHover={{ y: -2 }}
                        transition={{ duration: 0.2 }}
                    >
                        {/* Avatar Section */}
                        <div className="p-8 text-center relative overflow-hidden">
                            <div
                                className="absolute inset-0"
                                animate={
                                    {
                                        // background: [
                                        //     "linear-gradient(45deg, rgba(59, 130, 246, 0.05) 0%, transparent 100%)",
                                        //     "linear-gradient(135deg, rgba(59, 130, 246, 0.05) 0%, transparent 100%)",
                                        //     "linear-gradient(45deg, rgba(59, 130, 246, 0.05) 0%, transparent 100%)",
                                        // ],
                                    }
                                }
                                transition={{ duration: 4, repeat: Infinity }}
                            />

                            <div className="relative">
                                <motion.div
                                    className="relative inline-block"
                                    whileHover={{ scale: 1.05 }}
                                    transition={{ duration: 0.2 }}
                                >
                                    {user.avatar ? (
                                        <motion.img
                                            src={user.avatar}
                                            alt="Profile Avatar"
                                            className="w-40 h-40 rounded-full border-4 border-white/50 dark:border-dark-background/50 shadow-2xl object-cover backdrop-blur-sm"
                                            initial={{ scale: 0.8, opacity: 0 }}
                                            animate={{ scale: 1, opacity: 1 }}
                                            transition={{ duration: 0.5 }}
                                            onError={(e) => {
                                                e.target.style.display = "none";
                                                e.target.nextSibling.style.display =
                                                    "flex";
                                            }}
                                        />
                                    ) : null}
                                    <motion.div
                                        className={`w-32 h-32 rounded-full border-4 border-white/50 dark:border-dark-background/50 shadow-2xl bg-gradient-to-br from-light-highlight to-light-highlight/80 dark:from-dark-highlight dark:to-dark-highlight/80 flex items-center justify-center text-white text-4xl font-bold backdrop-blur-sm ${
                                            user.avatar ? "hidden" : "flex"
                                        }`}
                                        initial={{ scale: 0.8, opacity: 0 }}
                                        animate={{ scale: 1, opacity: 1 }}
                                        transition={{ duration: 0.5 }}
                                    >
                                        {user.firstName
                                            ?.charAt(0)
                                            ?.toUpperCase() || "U"}
                                    </motion.div>

                                    <Button
                                        onClick={() => setAvatarMode(true)}
                                        className="absolute -bottom-2 -right-2 p-3 rounded-full shadow-lg hover:shadow-xl backdrop-blur-sm border-2 border-white/20"
                                        variant="default"
                                        size="sm"
                                    >
                                        <Camera className="w-5 h-5" />
                                    </Button>
                                </motion.div>

                                <motion.div
                                    className="mt-6"
                                    initial={{ y: 20, opacity: 0 }}
                                    animate={{ y: 0, opacity: 1 }}
                                    transition={{ delay: 0.3, duration: 0.5 }}
                                >
                                    <h2 className="text-2xl font-bold text-light-text dark:text-dark-text">
                                        {user.firstName} {user.lastName}
                                    </h2>
                                    <p className="text-light-muted-text dark:text-dark-muted-text mt-1">
                                        {user.email}
                                    </p>
                                    <motion.div
                                        className="flex items-center justify-center mt-3"
                                        whileHover={{ scale: 1.05 }}
                                    >
                                        <div className="flex items-center px-3 py-1.5 bg-light-highlight/10 dark:bg-dark-highlight/10 rounded-full border border-light-highlight/20 dark:border-dark-highlight/20">
                                            <Shield className="w-4 h-4 mr-2 text-light-highlight dark:text-dark-highlight" />
                                            <span className="text-sm text-light-highlight dark:text-dark-highlight font-medium capitalize">
                                                {user.role?.toLowerCase() ||
                                                    "User"}
                                            </span>
                                        </div>
                                    </motion.div>
                                </motion.div>
                            </div>
                        </div>
                    </motion.div>

                    {/* Account Info Card */}
                    <motion.div
                        className="bg-light-background dark:bg-dark-noisy-background backdrop-blur-xl border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-2xl shadow-xl p-6"
                        whileHover={{ y: -2 }}
                        transition={{ duration: 0.2 }}
                    >
                        <motion.div
                            className="flex items-center text-sm text-light-muted-text dark:text-dark-muted-text"
                            initial={{ x: -20, opacity: 0 }}
                            animate={{ x: 0, opacity: 1 }}
                            transition={{ delay: 0.4, duration: 0.5 }}
                        >
                            <Calendar className="w-5 h-5 mr-3 text-light-highlight dark:text-dark-highlight" />
                            <span>
                                Member since{" "}
                                <span className="font-medium text-light-text dark:text-dark-text">
                                    {new Date(
                                        user.createdAt
                                    ).toLocaleDateString("en-US", {
                                        year: "numeric",
                                        month: "long",
                                        day: "numeric",
                                    })}
                                </span>
                            </span>
                        </motion.div>
                    </motion.div>
                </motion.div>

                {/* Right Column - Profile Information */}
                <motion.div className="xl:col-span-2" variants={cardVariants}>
                    <motion.div
                        className="bg-light-background dark:bg-dark-noisy-background backdrop-blur-xl border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-2xl shadow-xl overflow-hidden"
                        whileHover={{ y: -2 }}
                        transition={{ duration: 0.2 }}
                    >
                        {/* Profile Information */}
                        <div className="p-8 space-y-10">
                            {/* Basic Info Section */}
                            <div>
                                <div className="flex items-center justify-between mb-8">
                                    <motion.h3
                                        className="text-xl font-bold text-light-text dark:text-dark-text flex items-center"
                                        initial={{ x: -20, opacity: 0 }}
                                        animate={{ x: 0, opacity: 1 }}
                                        transition={{
                                            delay: 0.2,
                                            duration: 0.5,
                                        }}
                                    >
                                        <div className="p-2 bg-light-highlight/10 dark:bg-dark-highlight/10 rounded-lg mr-3">
                                            <User className="w-6 h-6 text-light-highlight dark:text-dark-highlight" />
                                        </div>
                                        Personal Information
                                    </motion.h3>
                                </div>

                                <AnimatePresence mode="wait">
                                    {editMode ? (
                                        /* Edit Mode */
                                        <motion.div
                                            className="space-y-8"
                                            initial={{ opacity: 0, x: 20 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            exit={{ opacity: 0, x: -20 }}
                                            transition={{ duration: 0.3 }}
                                        >
                                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                                <motion.div
                                                    initial={{
                                                        y: 20,
                                                        opacity: 0,
                                                    }}
                                                    animate={{
                                                        y: 0,
                                                        opacity: 1,
                                                    }}
                                                    transition={{
                                                        delay: 0.1,
                                                        duration: 0.3,
                                                    }}
                                                >
                                                    <label className="block text-sm font-semibold text-light-text dark:text-dark-text mb-2">
                                                        First Name
                                                    </label>
                                                    <input
                                                        type="text"
                                                        name="firstName"
                                                        value={
                                                            profileForm.firstName
                                                        }
                                                        onChange={
                                                            handleProfileChange
                                                        }
                                                        className="w-full px-4 py-3 bg-light-muted-background/50 dark:bg-dark-muted-background/50 border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-xl text-light-text dark:text-dark-text focus:outline-none focus:border-light-highlight dark:focus:border-dark-highlight focus:ring-2 focus:ring-light-highlight/20 dark:focus:ring-dark-highlight/20 transition-all duration-200 backdrop-blur-sm"
                                                    />
                                                </motion.div>
                                                <motion.div
                                                    initial={{
                                                        y: 20,
                                                        opacity: 0,
                                                    }}
                                                    animate={{
                                                        y: 0,
                                                        opacity: 1,
                                                    }}
                                                    transition={{
                                                        delay: 0.2,
                                                        duration: 0.3,
                                                    }}
                                                >
                                                    <label className="block text-sm font-semibold text-light-text dark:text-dark-text mb-2">
                                                        Last Name
                                                    </label>
                                                    <input
                                                        type="text"
                                                        name="lastName"
                                                        value={
                                                            profileForm.lastName
                                                        }
                                                        onChange={
                                                            handleProfileChange
                                                        }
                                                        className="w-full px-4 py-3 bg-light-muted-background/50 dark:bg-dark-muted-background/50 border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-xl text-light-text dark:text-dark-text focus:outline-none focus:border-light-highlight dark:focus:border-dark-highlight focus:ring-2 focus:ring-light-highlight/20 dark:focus:ring-dark-highlight/20 transition-all duration-200 backdrop-blur-sm"
                                                    />
                                                </motion.div>
                                            </div>

                                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                                <motion.div
                                                    initial={{
                                                        y: 20,
                                                        opacity: 0,
                                                    }}
                                                    animate={{
                                                        y: 0,
                                                        opacity: 1,
                                                    }}
                                                    transition={{
                                                        delay: 0.3,
                                                        duration: 0.3,
                                                    }}
                                                >
                                                    <label className="block text-sm font-semibold text-light-text dark:text-dark-text mb-2">
                                                        Email Address
                                                    </label>
                                                    <div className="w-full px-4 py-3 bg-light-muted-background/30 dark:bg-dark-muted-background/30 border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-xl text-light-muted-text dark:text-dark-muted-text cursor-not-allowed backdrop-blur-sm">
                                                        {user.email}
                                                    </div>
                                                    <p className="text-xs text-light-muted-text dark:text-dark-muted-text mt-2">
                                                        Email cannot be changed
                                                    </p>
                                                </motion.div>
                                                <motion.div
                                                    initial={{
                                                        y: 20,
                                                        opacity: 0,
                                                    }}
                                                    animate={{
                                                        y: 0,
                                                        opacity: 1,
                                                    }}
                                                    transition={{
                                                        delay: 0.4,
                                                        duration: 0.3,
                                                    }}
                                                >
                                                    <label className="block text-sm font-semibold text-light-text dark:text-dark-text mb-2">
                                                        Phone Number
                                                    </label>
                                                    <input
                                                        type="tel"
                                                        name="phone"
                                                        value={
                                                            profileForm.phone
                                                        }
                                                        onChange={
                                                            handleProfileChange
                                                        }
                                                        placeholder="+91 9876543210"
                                                        className="w-full px-4 py-3 bg-light-muted-background/50 dark:bg-dark-muted-background/50 border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-xl text-light-text dark:text-dark-text focus:outline-none focus:border-light-highlight dark:focus:border-dark-highlight focus:ring-2 focus:ring-light-highlight/20 dark:focus:ring-dark-highlight/20 transition-all duration-200 backdrop-blur-sm"
                                                    />
                                                    <p className="text-xs text-light-muted-text dark:text-dark-muted-text mt-2">
                                                        Format: +91 followed by
                                                        10-digit mobile number
                                                    </p>
                                                </motion.div>
                                            </div>

                                            <motion.div
                                                initial={{ y: 20, opacity: 0 }}
                                                animate={{ y: 0, opacity: 1 }}
                                                transition={{
                                                    delay: 0.5,
                                                    duration: 0.3,
                                                }}
                                            >
                                                <label className="block text-sm font-semibold text-light-text dark:text-dark-text mb-2">
                                                    Bio
                                                </label>
                                                <textarea
                                                    name="bio"
                                                    value={profileForm.bio}
                                                    onChange={
                                                        handleProfileChange
                                                    }
                                                    rows={4}
                                                    maxLength={500}
                                                    placeholder="Tell us about yourself..."
                                                    className="w-full px-4 py-3 bg-light-muted-background/50 dark:bg-dark-muted-background/50 border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-xl text-light-text dark:text-dark-text focus:outline-none focus:border-light-highlight dark:focus:border-dark-highlight focus:ring-2 focus:ring-light-highlight/20 dark:focus:ring-dark-highlight/20 transition-all duration-200 resize-none backdrop-blur-sm"
                                                />
                                                <p className="text-xs text-light-muted-text dark:text-dark-muted-text mt-2">
                                                    {profileForm.bio.length}/500
                                                    characters
                                                </p>
                                            </motion.div>

                                            <motion.div
                                                className="flex items-center justify-start space-x-4 pt-6"
                                                initial={{ y: 20, opacity: 0 }}
                                                animate={{ y: 0, opacity: 1 }}
                                                transition={{
                                                    delay: 0.6,
                                                    duration: 0.3,
                                                }}
                                            >
                                                <Button
                                                    onClick={updateProfile}
                                                    disabled={
                                                        updateProfileMutation.isPending
                                                    }
                                                    className="flex items-center space-x-2"
                                                    variant="default"
                                                >
                                                    <Save className="w-4 h-4" />
                                                    <span>
                                                        {updateProfileMutation.isPending
                                                            ? "Saving..."
                                                            : "Save Changes"}
                                                    </span>
                                                </Button>
                                                <Button
                                                    onClick={cancelEdit}
                                                    variant="outline"
                                                    className="flex items-center space-x-2"
                                                >
                                                    <X className="w-4 h-4" />
                                                    <span>Cancel</span>
                                                </Button>
                                            </motion.div>
                                        </motion.div>
                                    ) : (
                                        /* View Mode */
                                        <motion.div
                                            className="space-y-8"
                                            initial={{ opacity: 0, x: -20 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            transition={{ duration: 0.3 }}
                                        >
                                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                                                <motion.div
                                                    initial={{
                                                        y: 20,
                                                        opacity: 0,
                                                    }}
                                                    animate={{
                                                        y: 0,
                                                        opacity: 1,
                                                    }}
                                                    transition={{
                                                        delay: 0.1,
                                                        duration: 0.3,
                                                    }}
                                                >
                                                    <label className="block text-sm font-medium text-light-muted-text dark:text-dark-muted-text mb-3">
                                                        First Name
                                                    </label>
                                                    <p className="text-light-text dark:text-dark-text font-semibold text-lg">
                                                        {user.firstName ||
                                                            "Not provided"}
                                                    </p>
                                                </motion.div>
                                                <motion.div
                                                    initial={{
                                                        y: 20,
                                                        opacity: 0,
                                                    }}
                                                    animate={{
                                                        y: 0,
                                                        opacity: 1,
                                                    }}
                                                    transition={{
                                                        delay: 0.2,
                                                        duration: 0.3,
                                                    }}
                                                >
                                                    <label className="block text-sm font-medium text-light-muted-text dark:text-dark-muted-text mb-3">
                                                        Last Name
                                                    </label>
                                                    <p className="text-light-text dark:text-dark-text font-semibold text-lg">
                                                        {user.lastName ||
                                                            "Not provided"}
                                                    </p>
                                                </motion.div>
                                            </div>

                                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                                                <motion.div
                                                    initial={{
                                                        y: 20,
                                                        opacity: 0,
                                                    }}
                                                    animate={{
                                                        y: 0,
                                                        opacity: 1,
                                                    }}
                                                    transition={{
                                                        delay: 0.3,
                                                        duration: 0.3,
                                                    }}
                                                >
                                                    <label className="block text-sm font-medium text-light-muted-text dark:text-dark-muted-text mb-3">
                                                        Email Address
                                                    </label>
                                                    <div className="flex items-center space-x-3">
                                                        <div className="p-2 bg-light-highlight/10 dark:bg-dark-highlight/10 rounded-lg">
                                                            <Mail className="w-5 h-5 text-light-highlight dark:text-dark-highlight" />
                                                        </div>
                                                        <p className="text-light-text dark:text-dark-text font-semibold text-lg">
                                                            {user.email}
                                                        </p>
                                                    </div>
                                                </motion.div>
                                                <motion.div
                                                    initial={{
                                                        y: 20,
                                                        opacity: 0,
                                                    }}
                                                    animate={{
                                                        y: 0,
                                                        opacity: 1,
                                                    }}
                                                    transition={{
                                                        delay: 0.4,
                                                        duration: 0.3,
                                                    }}
                                                >
                                                    <label className="block text-sm font-medium text-light-muted-text dark:text-dark-muted-text mb-3">
                                                        Phone Number
                                                    </label>
                                                    <div className="flex items-center space-x-3">
                                                        <div className="p-2 bg-light-highlight/10 dark:bg-dark-highlight/10 rounded-lg">
                                                            <Phone className="w-5 h-5 text-light-highlight dark:text-dark-highlight" />
                                                        </div>
                                                        <p className="text-light-text dark:text-dark-text font-semibold text-lg">
                                                            {user.phone ||
                                                                "Not provided"}
                                                        </p>
                                                    </div>
                                                </motion.div>
                                            </div>

                                            <motion.div
                                                initial={{ y: 20, opacity: 0 }}
                                                animate={{ y: 0, opacity: 1 }}
                                                transition={{
                                                    delay: 0.5,
                                                    duration: 0.3,
                                                }}
                                            >
                                                <label className="block text-sm font-medium text-light-muted-text dark:text-dark-muted-text mb-3">
                                                    Bio
                                                </label>
                                                <div className="p-4 bg-light-muted-background/30 dark:bg-dark-muted-background/30 rounded-xl border border-light-muted-text/10 dark:border-dark-muted-text/10">
                                                    <p className="text-light-text dark:text-dark-text text-lg leading-relaxed">
                                                        {user.bio ||
                                                            "No bio provided"}
                                                    </p>
                                                </div>
                                            </motion.div>
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </div>

                            {/* Password Change Section */}
                            <AnimatePresence>
                                {passwordMode && (
                                    <motion.div
                                        initial={{ opacity: 0, y: 20 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        exit={{ opacity: 0, y: -20 }}
                                        transition={{ duration: 0.3 }}
                                        className="border-t border-light-muted-text/20 dark:border-dark-muted-text/20 pt-8"
                                    >
                                        <div className="flex items-center justify-between mb-8">
                                            <motion.h3
                                                className="text-xl font-bold text-light-text dark:text-dark-text flex items-center"
                                                initial={{ x: -20, opacity: 0 }}
                                                animate={{ x: 0, opacity: 1 }}
                                                transition={{
                                                    delay: 0.2,
                                                    duration: 0.5,
                                                }}
                                            >
                                                <div className="p-2 bg-light-highlight/10 dark:bg-dark-highlight/10 rounded-lg mr-3">
                                                    <Lock className="w-6 h-6 text-light-highlight dark:text-dark-highlight" />
                                                </div>
                                                Change Password
                                            </motion.h3>
                                        </div>

                                        <motion.div
                                            className="space-y-6"
                                            initial={{ opacity: 0, x: 20 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            exit={{ opacity: 0, x: -20 }}
                                            transition={{ duration: 0.3 }}
                                        >
                                            <motion.div
                                                initial={{ y: 20, opacity: 0 }}
                                                animate={{ y: 0, opacity: 1 }}
                                                transition={{
                                                    delay: 0.1,
                                                    duration: 0.3,
                                                }}
                                            >
                                                <label className="block text-sm font-semibold text-light-text dark:text-dark-text mb-2">
                                                    Current Password
                                                </label>
                                                <input
                                                    type="password"
                                                    name="currentPassword"
                                                    value={
                                                        passwordForm.currentPassword
                                                    }
                                                    onChange={
                                                        handlePasswordChange
                                                    }
                                                    className="w-full px-4 py-3 bg-light-muted-background/50 dark:bg-dark-muted-background/50 border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-xl text-light-text dark:text-dark-text focus:outline-none focus:border-light-highlight dark:focus:border-dark-highlight focus:ring-2 focus:ring-light-highlight/20 dark:focus:ring-dark-highlight/20 transition-all duration-200 backdrop-blur-sm"
                                                    placeholder="Enter your current password"
                                                />
                                            </motion.div>

                                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                                <motion.div
                                                    initial={{
                                                        y: 20,
                                                        opacity: 0,
                                                    }}
                                                    animate={{
                                                        y: 0,
                                                        opacity: 1,
                                                    }}
                                                    transition={{
                                                        delay: 0.2,
                                                        duration: 0.3,
                                                    }}
                                                >
                                                    <label className="block text-sm font-semibold text-light-text dark:text-dark-text mb-2">
                                                        New Password
                                                    </label>
                                                    <input
                                                        type="password"
                                                        name="newPassword"
                                                        value={
                                                            passwordForm.newPassword
                                                        }
                                                        onChange={
                                                            handlePasswordChange
                                                        }
                                                        className="w-full px-4 py-3 bg-light-muted-background/50 dark:bg-dark-muted-background/50 border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-xl text-light-text dark:text-dark-text focus:outline-none focus:border-light-highlight dark:focus:border-dark-highlight focus:ring-2 focus:ring-light-highlight/20 dark:focus:ring-dark-highlight/20 transition-all duration-200 backdrop-blur-sm"
                                                        placeholder="Enter new password"
                                                    />
                                                </motion.div>

                                                <motion.div
                                                    initial={{
                                                        y: 20,
                                                        opacity: 0,
                                                    }}
                                                    animate={{
                                                        y: 0,
                                                        opacity: 1,
                                                    }}
                                                    transition={{
                                                        delay: 0.3,
                                                        duration: 0.3,
                                                    }}
                                                >
                                                    <label className="block text-sm font-semibold text-light-text dark:text-dark-text mb-2">
                                                        Confirm New Password
                                                    </label>
                                                    <input
                                                        type="password"
                                                        name="confirmPassword"
                                                        value={
                                                            passwordForm.confirmPassword
                                                        }
                                                        onChange={
                                                            handlePasswordChange
                                                        }
                                                        className="w-full px-4 py-3 bg-light-muted-background/50 dark:bg-dark-muted-background/50 border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-xl text-light-text dark:text-dark-text focus:outline-none focus:border-light-highlight dark:focus:border-dark-highlight focus:ring-2 focus:ring-light-highlight/20 dark:focus:ring-dark-highlight/20 transition-all duration-200 backdrop-blur-sm"
                                                        placeholder="Confirm new password"
                                                    />
                                                </motion.div>
                                            </div>

                                            <motion.div
                                                className="flex items-center justify-start space-x-4 pt-6"
                                                initial={{ y: 20, opacity: 0 }}
                                                animate={{ y: 0, opacity: 1 }}
                                                transition={{
                                                    delay: 0.4,
                                                    duration: 0.3,
                                                }}
                                            >
                                                <Button
                                                    onClick={updatePassword}
                                                    disabled={
                                                        updatePasswordMutation.isPending ||
                                                        !passwordForm.currentPassword ||
                                                        !passwordForm.newPassword ||
                                                        !passwordForm.confirmPassword
                                                    }
                                                    className="flex items-center space-x-2"
                                                    variant="default"
                                                >
                                                    <Save className="w-4 h-4" />
                                                    <span>
                                                        {updatePasswordMutation.isPending
                                                            ? "Updating..."
                                                            : "Update Password"}
                                                    </span>
                                                </Button>
                                                <Button
                                                    onClick={() => {
                                                        setPasswordMode(false);
                                                        setPasswordForm({
                                                            currentPassword: "",
                                                            newPassword: "",
                                                            confirmPassword: "",
                                                        });
                                                    }}
                                                    variant="outline"
                                                    className="flex items-center space-x-2"
                                                >
                                                    <X className="w-4 h-4" />
                                                    <span>Cancel</span>
                                                </Button>
                                            </motion.div>
                                        </motion.div>
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </div>
                    </motion.div>
                </motion.div>
            </div>

            {/* Avatar Modal */}
            <AnimatePresence>
                {avatarMode && (
                    <motion.div
                        className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-[99999] p-4"
                        variants={overlayVariants}
                        initial="hidden"
                        animate="visible"
                        exit="exit"
                        onClick={cancelAvatar}
                    >
                        <motion.div
                            className="bg-light-background dark:bg-dark-noisy-background backdrop-blur-xl rounded-2xl p-8 w-full max-w-2xl mx-4 border border-light-muted-text/20 dark:border-dark-muted-text/20 shadow-2xl"
                            variants={modalVariants}
                            initial="hidden"
                            animate="visible"
                            exit="exit"
                            onClick={(e) => e.stopPropagation()}
                        >
                            <div className="flex items-center justify-between mb-6">
                                <motion.h3
                                    className="text-xl font-bold text-light-text dark:text-dark-text text-center"
                                    initial={{ y: -10, opacity: 0 }}
                                    animate={{ y: 0, opacity: 1 }}
                                    transition={{ delay: 0.1, duration: 0.3 }}
                                >
                                    Update Avatar
                                </motion.h3>

                                <Button
                                    onClick={cancelAvatar}
                                    className="flex items-center justify-center"
                                    variant="outline"
                                    size="sm"
                                >
                                    <X className="w-4 h-4" />
                                </Button>
                            </div>

                            <motion.div
                                className="space-y-6"
                                initial={{ y: 20, opacity: 0 }}
                                animate={{ y: 0, opacity: 1 }}
                                transition={{ delay: 0.2, duration: 0.3 }}
                            >
                                <div>
                                    <label className="block text-sm font-semibold text-light-text dark:text-dark-text mb-2">
                                        Avatar URL
                                    </label>
                                    <input
                                        type="url"
                                        name="avatar"
                                        value={avatarForm.avatar}
                                        onChange={handleAvatarChange}
                                        placeholder="https://example.com/avatar.jpg"
                                        className="w-full px-4 py-3 bg-light-muted-background/50 dark:bg-dark-muted-background/50 border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-xl text-light-text dark:text-dark-text focus:outline-none focus:border-light-highlight dark:focus:border-dark-highlight focus:ring-2 focus:ring-light-highlight/20 dark:focus:ring-dark-highlight/20 transition-all duration-200 backdrop-blur-sm"
                                    />
                                </div>

                                {avatarForm.avatar && (
                                    <motion.div
                                        className="text-center"
                                        initial={{ scale: 0.8, opacity: 0 }}
                                        animate={{ scale: 1, opacity: 1 }}
                                        transition={{ duration: 0.3 }}
                                    >
                                        <img
                                            src={avatarForm.avatar}
                                            alt="Avatar Preview"
                                            className="w-40 h-40 rounded-full mx-auto object-cover border-4 border-light-highlight/20 dark:border-dark-highlight/20 shadow-lg"
                                            onError={(e) => {
                                                e.target.style.display = "none";
                                            }}
                                        />
                                    </motion.div>
                                )}

                                <div className="flex items-center justify-between space-x-3">
                                    <div className="w-full flex items-center justify-between space-x-3">
                                        <Button
                                            onClick={updateAvatar}
                                            disabled={
                                                !avatarForm.avatar ||
                                                updateAvatarMutation.isPending
                                            }
                                            className="flex items-center justify-center space-x-2"
                                            variant="default"
                                        >
                                            <Save className="w-5 h-5" />
                                            <span>
                                                {updateAvatarMutation.isPending
                                                    ? "Saving..."
                                                    : "Save"}
                                            </span>
                                        </Button>

                                        {user.avatar && (
                                            <Button
                                                onClick={deleteAvatar}
                                                disabled={
                                                    deleteAvatarMutation.isPending
                                                }
                                                className="flex items-center justify-center space-x-2"
                                                variant="destructive"
                                            >
                                                <Trash2 className="w-4 h-4" />
                                                <span>
                                                    {deleteAvatarMutation.isPending
                                                        ? "Removing..."
                                                        : "Remove"}
                                                </span>
                                            </Button>
                                        )}
                                    </div>
                                </div>
                            </motion.div>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.div>
    );
};

export default Profile;
