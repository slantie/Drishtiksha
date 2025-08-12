import React from "react";
import {
    User,
    Mail,
    Phone,
    Edit3,
    Save,
    X,
    Camera,
    Lock,
    Eye,
    EyeOff,
    Trash2,
    Calendar,
    Shield,
} from "lucide-react";
import { PageLoader } from "../components/ui/LoadingSpinner";
import { useProfile } from "../hooks/useProfile";

const Profile = () => {
    const {
        user,
        loading,
        editMode,
        passwordMode,
        avatarMode,
        profileForm,
        passwordForm,
        avatarForm,
        showPasswords,
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
    } = useProfile();

    if (loading) {
        return <PageLoader text="Profile Loading" />;
    }

    if (!user) {
        return (
            <div className="w-full mx-auto p-4 lg:p-6">
                <div className="text-center py-16">
                    <h2 className="text-2xl font-bold text-light-text dark:text-dark-text mb-4">
                        Profile Not Found
                    </h2>
                    <p className="text-light-muted-text dark:text-dark-muted-text text-lg">
                        Unable to load your profile. Please try logging in
                        again.
                    </p>
                </div>
            </div>
        );
    }

    return (
        <div className="w-full mx-auto p-4 lg:p-6">
            {/* Header */}
            <div className="text-center mb-6 lg:mb-8">
                <h1 className="text-2xl lg:text-3xl font-bold text-light-text dark:text-dark-text mb-2">
                    My Profile
                </h1>
                <p className="text-light-muted-text dark:text-dark-muted-text">
                    Manage your personal information and account settings
                </p>
            </div>

            {/* Main Layout Grid */}
            <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 lg:gap-8">
                {/* Left Column - Avatar & Quick Info */}
                <div className="xl:col-span-1 space-y-6">
                    {/* Avatar Card */}
                    <div className="bg-light-background dark:bg-dark-background border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-lg shadow-lg overflow-hidden">
                        {/* Avatar Section */}
                        <div className="bg-gradient-to-r from-light-highlight/10 to-light-highlight/5 dark:from-dark-highlight/10 dark:to-dark-highlight/5 p-6 text-center">
                            <div className="relative inline-block">
                                {user.avatar ? (
                                    <img
                                        src={user.avatar}
                                        alt="Profile Avatar"
                                        className="w-24 h-24 rounded-full border-4 border-white dark:border-dark-background shadow-lg object-cover"
                                        onError={(e) => {
                                            e.target.style.display = "none";
                                            e.target.nextSibling.style.display =
                                                "flex";
                                        }}
                                    />
                                ) : null}
                                <div
                                    className={`w-24 h-24 rounded-full border-4 border-white dark:border-dark-background shadow-lg bg-gradient-to-br from-light-highlight to-light-highlight/80 dark:from-dark-highlight dark:to-dark-highlight/80 flex items-center justify-center text-white text-2xl font-bold ${
                                        user.avatar ? "hidden" : "flex"
                                    }`}
                                >
                                    {user.firstName?.charAt(0)?.toUpperCase() ||
                                        "U"}
                                </div>

                                <button
                                    onClick={() => setAvatarMode(true)}
                                    className="absolute -bottom-2 -right-2 bg-light-highlight dark:bg-dark-highlight text-white p-2 rounded-full shadow-lg hover:scale-110 transition-transform duration-200"
                                >
                                    <Camera className="w-4 h-4" />
                                </button>
                            </div>

                            <h2 className="text-xl font-semibold text-light-text dark:text-dark-text mt-4">
                                {user.firstName} {user.lastName}
                            </h2>
                            <p className="text-light-muted-text dark:text-dark-muted-text">
                                {user.email}
                            </p>
                            <div className="flex items-center justify-center mt-2">
                                <Shield className="w-4 h-4 mr-1 text-light-highlight dark:text-dark-highlight" />
                                <span className="text-sm text-light-highlight dark:text-dark-highlight font-medium capitalize">
                                    {user.role?.toLowerCase() || "User"}
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* Account Info Card */}
                    <div className="bg-light-background dark:bg-dark-background border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-lg shadow-lg p-6">
                        <div className="flex items-center text-sm text-light-muted-text dark:text-dark-muted-text">
                            <Calendar className="w-4 h-4 mr-2" />
                            <span>
                                Account created:{" "}
                                {new Date(user.createdAt).toLocaleDateString()}
                            </span>
                        </div>
                    </div>
                </div>

                {/* Right Column - Profile Information */}
                <div className="xl:col-span-2">
                    <div className="bg-light-background dark:bg-dark-background border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-lg shadow-lg overflow-hidden">
                        {/* Profile Information */}
                        <div className="p-6 space-y-8">
                            {/* Basic Info Section */}
                            <div>
                                <div className="flex items-center justify-between mb-6">
                                    <h3 className="text-lg font-semibold text-light-text dark:text-dark-text flex items-center">
                                        <User className="w-5 h-5 mr-2 text-light-highlight dark:text-dark-highlight" />
                                        Personal Information
                                    </h3>
                                    {!editMode && (
                                        <button
                                            onClick={() => setEditMode(true)}
                                            className="flex items-center space-x-1 text-light-highlight dark:text-dark-highlight hover:bg-light-muted-background dark:hover:bg-dark-muted-background px-3 py-2 rounded-lg transition-colors"
                                        >
                                            <Edit3 className="w-4 h-4" />
                                            <span>Edit</span>
                                        </button>
                                    )}
                                </div>

                                {editMode ? (
                                    /* Edit Mode */
                                    <div className="space-y-6">
                                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                            <div>
                                                <label className="block text-sm font-medium text-light-text dark:text-dark-text mb-1">
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
                                                    className="w-full px-4 py-3 bg-light-muted-background dark:bg-dark-muted-background border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-lg text-light-text dark:text-dark-text focus:outline-none focus:border-light-highlight dark:focus:border-dark-highlight transition-colors"
                                                />
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-light-text dark:text-dark-text mb-1">
                                                    Last Name
                                                </label>
                                                <input
                                                    type="text"
                                                    name="lastName"
                                                    value={profileForm.lastName}
                                                    onChange={
                                                        handleProfileChange
                                                    }
                                                    className="w-full px-4 py-3 bg-light-muted-background dark:bg-dark-muted-background border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-lg text-light-text dark:text-dark-text focus:outline-none focus:border-light-highlight dark:focus:border-dark-highlight transition-colors"
                                                />
                                            </div>
                                        </div>

                                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                            <div>
                                                <label className="block text-sm font-medium text-light-text dark:text-dark-text mb-1">
                                                    Email Address
                                                </label>
                                                <div className="w-full px-4 py-3 bg-light-muted-background/50 dark:bg-dark-muted-background/50 border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-lg text-light-muted-text dark:text-dark-muted-text cursor-not-allowed">
                                                    {user.email}
                                                </div>
                                                <p className="text-xs text-light-muted-text dark:text-dark-muted-text mt-1">
                                                    Email cannot be changed
                                                </p>
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-light-text dark:text-dark-text mb-1">
                                                    Phone Number
                                                </label>
                                                <input
                                                    type="tel"
                                                    name="phone"
                                                    value={profileForm.phone}
                                                    onChange={
                                                        handleProfileChange
                                                    }
                                                    placeholder="+91 9876543210"
                                                    className="w-full px-4 py-3 bg-light-muted-background dark:bg-dark-muted-background border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-lg text-light-text dark:text-dark-text focus:outline-none focus:border-light-highlight dark:focus:border-dark-highlight transition-colors"
                                                />
                                                <p className="text-xs text-light-muted-text dark:text-dark-muted-text mt-1">
                                                    Format: +91 followed by
                                                    10-digit mobile number
                                                </p>
                                            </div>
                                        </div>

                                        <div>
                                            <label className="block text-sm font-medium text-light-text dark:text-dark-text mb-1">
                                                Bio
                                            </label>
                                            <textarea
                                                name="bio"
                                                value={profileForm.bio}
                                                onChange={handleProfileChange}
                                                rows={4}
                                                placeholder="Tell us about yourself..."
                                                className="w-full px-4 py-3 bg-light-muted-background dark:bg-dark-muted-background border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-lg text-light-text dark:text-dark-text focus:outline-none focus:border-light-highlight dark:focus:border-dark-highlight transition-colors resize-none"
                                            />
                                            <p className="text-xs text-light-muted-text dark:text-dark-muted-text mt-1">
                                                {profileForm.bio.length}/500
                                                characters
                                            </p>
                                        </div>

                                        <div className="flex items-center justify-start space-x-4 pt-4">
                                            <button
                                                onClick={updateProfile}
                                                className="flex items-center space-x-2 bg-light-highlight dark:bg-dark-highlight text-white px-6 py-3 rounded-lg hover:opacity-90 transition-opacity font-medium"
                                            >
                                                <Save className="w-4 h-4" />
                                                <span>Save Changes</span>
                                            </button>
                                            <button
                                                onClick={cancelEdit}
                                                className="flex items-center space-x-2 bg-light-muted-background dark:bg-dark-muted-background text-light-text dark:text-dark-text px-6 py-3 rounded-lg hover:bg-light-muted-background/80 dark:hover:bg-dark-muted-background/80 transition-colors font-medium"
                                            >
                                                <X className="w-4 h-4" />
                                                <span>Cancel</span>
                                            </button>
                                        </div>
                                    </div>
                                ) : (
                                    /* View Mode */
                                    <div className="space-y-6">
                                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                                            <div>
                                                <label className="block text-sm font-medium text-light-muted-text dark:text-dark-muted-text mb-2">
                                                    First Name
                                                </label>
                                                <p className="text-light-text dark:text-dark-text font-medium text-lg">
                                                    {user.firstName ||
                                                        "Not provided"}
                                                </p>
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-light-muted-text dark:text-dark-muted-text mb-2">
                                                    Last Name
                                                </label>
                                                <p className="text-light-text dark:text-dark-text font-medium text-lg">
                                                    {user.lastName ||
                                                        "Not provided"}
                                                </p>
                                            </div>
                                        </div>

                                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                                            <div>
                                                <label className="block text-sm font-medium text-light-muted-text dark:text-dark-muted-text mb-2">
                                                    Email Address
                                                </label>
                                                <div className="flex items-center">
                                                    <Mail className="w-5 h-5 mr-3 text-light-muted-text dark:text-dark-muted-text" />
                                                    <p className="text-light-text dark:text-dark-text font-medium text-lg">
                                                        {user.email}
                                                    </p>
                                                </div>
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-light-muted-text dark:text-dark-muted-text mb-2">
                                                    Phone Number
                                                </label>
                                                <div className="flex items-center">
                                                    <Phone className="w-5 h-5 mr-3 text-light-muted-text dark:text-dark-muted-text" />
                                                    <p className="text-light-text dark:text-dark-text font-medium text-lg">
                                                        {user.phone ||
                                                            "Not provided"}
                                                    </p>
                                                </div>
                                            </div>
                                        </div>

                                        <div>
                                            <label className="block text-sm font-medium text-light-muted-text dark:text-dark-muted-text mb-2">
                                                Bio
                                            </label>
                                            <p className="text-light-text dark:text-dark-text text-lg leading-relaxed">
                                                {user.bio || "No bio provided"}
                                            </p>
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Account Security Section */}
                            <div className="border-t border-light-muted-text/20 dark:border-dark-muted-text/20 pt-8">
                                <div className="flex items-center justify-between mb-6">
                                    <h3 className="text-lg font-semibold text-light-text dark:text-dark-text flex items-center">
                                        <Lock className="w-5 h-5 mr-2 text-light-highlight dark:text-dark-highlight" />
                                        Account Security
                                    </h3>
                                    {!passwordMode && (
                                        <button
                                            onClick={() =>
                                                setPasswordMode(true)
                                            }
                                            className="flex items-center space-x-1 text-light-highlight dark:text-dark-highlight hover:bg-light-muted-background dark:hover:bg-dark-muted-background px-3 py-2 rounded-lg transition-colors"
                                        >
                                            <Edit3 className="w-4 h-4" />
                                            <span>Change Password</span>
                                        </button>
                                    )}
                                </div>

                                {passwordMode ? (
                                    /* Password Change Mode */
                                    <div className="space-y-6">
                                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                                            <div>
                                                <label className="block text-sm font-medium text-light-text dark:text-dark-text mb-1">
                                                    Current Password
                                                </label>
                                                <div className="relative">
                                                    <input
                                                        type={
                                                            showPasswords.current
                                                                ? "text"
                                                                : "password"
                                                        }
                                                        name="currentPassword"
                                                        value={
                                                            passwordForm.currentPassword
                                                        }
                                                        onChange={
                                                            handlePasswordChange
                                                        }
                                                        className="w-full px-4 py-3 pr-10 bg-light-muted-background dark:bg-dark-muted-background border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-lg text-light-text dark:text-dark-text focus:outline-none focus:border-light-highlight dark:focus:border-dark-highlight transition-colors"
                                                    />
                                                    <button
                                                        type="button"
                                                        onClick={() =>
                                                            togglePasswordVisibility(
                                                                "current"
                                                            )
                                                        }
                                                        className="absolute inset-y-0 right-0 pr-3 flex items-center text-light-muted-text dark:text-dark-muted-text hover:text-light-highlight dark:hover:text-dark-highlight"
                                                    >
                                                        {showPasswords.current ? (
                                                            <EyeOff className="w-4 h-4" />
                                                        ) : (
                                                            <Eye className="w-4 h-4" />
                                                        )}
                                                    </button>
                                                </div>
                                            </div>

                                            <div>
                                                <label className="block text-sm font-medium text-light-text dark:text-dark-text mb-1">
                                                    New Password
                                                </label>
                                                <div className="relative">
                                                    <input
                                                        type={
                                                            showPasswords.new
                                                                ? "text"
                                                                : "password"
                                                        }
                                                        name="newPassword"
                                                        value={
                                                            passwordForm.newPassword
                                                        }
                                                        onChange={
                                                            handlePasswordChange
                                                        }
                                                        className="w-full px-4 py-3 pr-10 bg-light-muted-background dark:bg-dark-muted-background border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-lg text-light-text dark:text-dark-text focus:outline-none focus:border-light-highlight dark:focus:border-dark-highlight transition-colors"
                                                    />
                                                    <button
                                                        type="button"
                                                        onClick={() =>
                                                            togglePasswordVisibility(
                                                                "new"
                                                            )
                                                        }
                                                        className="absolute inset-y-0 right-0 pr-3 flex items-center text-light-muted-text dark:text-dark-muted-text hover:text-light-highlight dark:hover:text-dark-highlight"
                                                    >
                                                        {showPasswords.new ? (
                                                            <EyeOff className="w-4 h-4" />
                                                        ) : (
                                                            <Eye className="w-4 h-4" />
                                                        )}
                                                    </button>
                                                </div>
                                            </div>

                                            <div>
                                                <label className="block text-sm font-medium text-light-text dark:text-dark-text mb-1">
                                                    Confirm New Password
                                                </label>
                                                <div className="relative">
                                                    <input
                                                        type={
                                                            showPasswords.confirm
                                                                ? "text"
                                                                : "password"
                                                        }
                                                        name="confirmPassword"
                                                        value={
                                                            passwordForm.confirmPassword
                                                        }
                                                        onChange={
                                                            handlePasswordChange
                                                        }
                                                        className="w-full px-4 py-3 pr-10 bg-light-muted-background dark:bg-dark-muted-background border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-lg text-light-text dark:text-dark-text focus:outline-none focus:border-light-highlight dark:focus:border-dark-highlight transition-colors"
                                                    />
                                                    <button
                                                        type="button"
                                                        onClick={() =>
                                                            togglePasswordVisibility(
                                                                "confirm"
                                                            )
                                                        }
                                                        className="absolute inset-y-0 right-0 pr-3 flex items-center text-light-muted-text dark:text-dark-muted-text hover:text-light-highlight dark:hover:text-dark-highlight"
                                                    >
                                                        {showPasswords.confirm ? (
                                                            <EyeOff className="w-4 h-4" />
                                                        ) : (
                                                            <Eye className="w-4 h-4" />
                                                        )}
                                                    </button>
                                                </div>
                                            </div>
                                        </div>

                                        <div className="flex items-center justify-start space-x-4 pt-4">
                                            <button
                                                onClick={updatePassword}
                                                className="flex items-center space-x-2 bg-light-highlight dark:bg-dark-highlight text-white px-6 py-3 rounded-lg hover:opacity-90 transition-opacity font-medium"
                                            >
                                                <Save className="w-4 h-4" />
                                                <span>Update Password</span>
                                            </button>
                                            <button
                                                onClick={cancelPassword}
                                                className="flex items-center space-x-2 bg-light-muted-background dark:bg-dark-muted-background text-light-text dark:text-dark-text px-6 py-3 rounded-lg hover:bg-light-muted-background/80 dark:hover:bg-dark-muted-background/80 transition-colors font-medium"
                                            >
                                                <X className="w-4 h-4" />
                                                <span>Cancel</span>
                                            </button>
                                        </div>
                                    </div>
                                ) : (
                                    /* Security Info View */
                                    <div>
                                        <p className="text-light-muted-text dark:text-dark-muted-text text-lg leading-relaxed">
                                            Keep your account secure by using a
                                            strong password and changing it
                                            regularly.
                                        </p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Avatar Modal */}
            {avatarMode && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                    <div className="bg-light-background dark:bg-dark-background rounded-lg p-6 w-full max-w-md mx-4">
                        <h3 className="text-lg font-semibold text-light-text dark:text-dark-text mb-4">
                            Update Avatar
                        </h3>

                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-light-text dark:text-dark-text mb-1">
                                    Avatar URL
                                </label>
                                <input
                                    type="url"
                                    name="avatar"
                                    value={avatarForm.avatar}
                                    onChange={handleAvatarChange}
                                    placeholder="https://example.com/avatar.jpg"
                                    className="w-full px-3 py-2 bg-light-muted-background dark:bg-dark-muted-background border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-lg text-light-text dark:text-dark-text focus:outline-none focus:border-light-highlight dark:focus:border-dark-highlight transition-colors"
                                />
                            </div>

                            {avatarForm.avatar && (
                                <div className="text-center">
                                    <img
                                        src={avatarForm.avatar}
                                        alt="Avatar Preview"
                                        className="w-20 h-20 rounded-full mx-auto object-cover border-2 border-light-muted-text/20 dark:border-dark-muted-text/20"
                                        onError={(e) => {
                                            e.target.style.display = "none";
                                        }}
                                    />
                                </div>
                            )}

                            <div className="flex items-center justify-between space-x-3">
                                <div className="flex space-x-3">
                                    <button
                                        onClick={updateAvatar}
                                        disabled={!avatarForm.avatar}
                                        className="flex items-center space-x-2 bg-light-highlight dark:bg-dark-highlight text-white px-4 py-2 rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        <Save className="w-4 h-4" />
                                        <span>Save</span>
                                    </button>
                                    <button
                                        onClick={cancelAvatar}
                                        className="flex items-center space-x-2 bg-light-muted-background dark:bg-dark-muted-background text-light-text dark:text-dark-text px-4 py-2 rounded-lg hover:bg-light-muted-background/80 dark:hover:bg-dark-muted-background/80 transition-colors"
                                    >
                                        <X className="w-4 h-4" />
                                        <span>Cancel</span>
                                    </button>
                                </div>

                                {user.avatar && (
                                    <button
                                        onClick={deleteAvatar}
                                        className="flex items-center space-x-2 bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition-colors"
                                    >
                                        <Trash2 className="w-4 h-4" />
                                        <span>Remove</span>
                                    </button>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default Profile;
