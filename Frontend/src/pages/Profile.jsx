// src/pages/Profile.jsx

import React, { useState, useEffect } from "react";
import {
    User,
    Mail,
    Phone,
    Save,
    X,
    Camera,
    Lock,
    Trash2,
    Calendar,
    Shield,
    Edit3,
    AlertCircle,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { PageLoader } from "../components/ui/LoadingSpinner";
import {
    useProfileQuery,
    useUpdateProfileMutation,
    useUpdatePasswordMutation,
    useUpdateAvatarMutation,
    useDeleteAvatarMutation,
} from "../hooks/useAuthQuery.js";
import {
    Card,
    CardContent,
    CardHeader,
    CardTitle,
    CardDescription,
    CardFooter,
} from "../components/ui/Card.jsx";
import { Button } from "../components/ui/Button.jsx";
import { Input } from "../components/ui/Input.jsx";
import { Modal } from "../components/ui/Modal.jsx";
import { PageHeader } from "../components/layout/PageHeader.jsx";

// Sub-component for the user's avatar and basic info.
const AvatarCard = ({ user, onAvatarClick }) => (
    <Card className="text-center">
        <CardContent className="p-8">
            <div className="relative inline-block">
                <img
                    src={
                        user.avatar ||
                        `https://ui-avatars.com/api/?name=${user.firstName}+${user.lastName}&background=random&color=fff`
                    }
                    alt="Profile Avatar"
                    className="w-32 h-32 rounded-full border-4 border-light-secondary dark:border-dark-secondary shadow-lg object-cover"
                />
                <Button
                    onClick={onAvatarClick}
                    size="icon"
                    className="absolute -bottom-2 -right-2 rounded-full"
                >
                    <Camera className="w-5 h-5" />
                </Button>
            </div>
            <h2 className="text-2xl font-bold mt-4">
                {user.firstName} {user.lastName}
            </h2>
            <p className="text-light-muted-text dark:text-dark-muted-text">
                {user.email}
            </p>
            <div className="mt-3 inline-flex items-center gap-2 px-3 py-1 bg-primary-main/10 text-primary-main rounded-full text-sm font-semibold capitalize">
                <Shield className="w-4 h-4" />{" "}
                {user.role?.toLowerCase() || "User"}
            </div>
        </CardContent>
    </Card>
);

// Sub-component for displaying user information fields.
const InfoField = ({ label, value, icon: Icon }) => (
    <div>
        <label className="text-sm font-medium text-light-muted-text dark:text-dark-muted-text">
            {label}
        </label>
        <div className="flex items-center gap-3 mt-1">
            {Icon && (
                <Icon className="h-5 w-5 text-primary-main flex-shrink-0" />
            )}
            <p className="text-base font-semibold truncate">
                {value || "Not provided"}
            </p>
        </div>
    </div>
);

const Profile = () => {
    // --- STATE AND LOGIC (PRESERVED) ---
    const { data: user, isLoading: isProfileLoading } = useProfileQuery();
    const updateProfileMutation = useUpdateProfileMutation();
    const updatePasswordMutation = useUpdatePasswordMutation();
    const updateAvatarMutation = useUpdateAvatarMutation();
    const deleteAvatarMutation = useDeleteAvatarMutation();

    const [isEditMode, setIsEditMode] = useState(false);
    const [isPasswordMode, setIsPasswordMode] = useState(false);
    const [isAvatarMode, setIsAvatarMode] = useState(false);

    const [profileForm, setProfileForm] = useState({
        firstName: "",
        lastName: "",
        phone: "",
        bio: "",
    });
    const [passwordForm, setPasswordForm] = useState({
        currentPassword: "",
        newPassword: "",
        confirmPassword: "",
    });
    const [avatarUrl, setAvatarUrl] = useState("");

    useEffect(() => {
        if (user) {
            setProfileForm({
                firstName: user.firstName || "",
                lastName: user.lastName || "",
                phone: user.phone || "",
                bio: user.bio || "",
            });
            setAvatarUrl(user.avatar || "");
        }
    }, [user]);

    // All handler functions are preserved...
    const handleProfileChange = (e) =>
        setProfileForm({ ...profileForm, [e.target.name]: e.target.value });
    const handlePasswordChange = (e) =>
        setPasswordForm({ ...passwordForm, [e.target.name]: e.target.value });
    const handleProfileSave = () => {
        // Ensure phone number is either null or a valid string, not empty.
        const dataToUpdate = {
            ...profileForm,
            phone: profileForm.phone || null,
        };
        updateProfileMutation.mutate(dataToUpdate, {
            onSuccess: () => setIsEditMode(false),
        });
    };
    const handlePasswordSave = () =>
        updatePasswordMutation.mutate(passwordForm, {
            onSuccess: () => {
                setIsPasswordMode(false);
                setPasswordForm({
                    currentPassword: "",
                    newPassword: "",
                    confirmPassword: "",
                });
            },
        });
    const handleAvatarSave = () =>
        updateAvatarMutation.mutate(
            { avatar: avatarUrl },
            { onSuccess: () => setIsAvatarMode(false) }
        );
    const handleAvatarDelete = () =>
        deleteAvatarMutation.mutate(undefined, {
            onSuccess: () => {
                setAvatarUrl("");
                setIsAvatarMode(false);
            },
        });

    // --- RENDER LOGIC ---
    if (isProfileLoading) return <PageLoader text="Loading Profile..." />;

    if (!user) {
        return (
            <div className="text-center py-16">
                <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
                <h2 className="text-2xl font-bold">Profile Not Found</h2>
                <p className="text-lg text-light-muted-text dark:text-dark-muted-text">
                    Could not load profile. Please try logging in again.
                </p>
            </div>
        );
    }

    // REFACTOR: The `as="textarea"` prop is a common pattern for component libraries to switch the underlying element.
    const Textarea = (props) => <Input as="textarea" {...props} />;

    return (
        <div className="space-y-8">
            <PageHeader
                title="Profile & Settings"
                description="Manage your personal information and account security."
                actions={
                    !isEditMode &&
                    !isPasswordMode && (
                        <div className="flex gap-2">
                            <Button
                                variant="outline"
                                onClick={() => setIsPasswordMode(true)}
                            >
                                <Lock className="mr-2 h-4 w-4" /> Change
                                Password
                            </Button>
                            <Button onClick={() => setIsEditMode(true)}>
                                <Edit3 className="mr-2 h-4 w-4" /> Edit Profile
                            </Button>
                        </div>
                    )
                }
            />

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Left Column */}
                <div className="lg:col-span-1 space-y-8">
                    <AvatarCard
                        user={user}
                        onAvatarClick={() => setIsAvatarMode(true)}
                    />
                    <Card>
                        <CardHeader>
                            <CardTitle>Account Info</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <InfoField
                                label="Member Since"
                                value={new Date(
                                    user.createdAt
                                ).toLocaleDateString()}
                                icon={Calendar}
                            />
                        </CardContent>
                    </Card>
                </div>

                {/* Right Column */}
                <div className="lg:col-span-2">
                    <AnimatePresence mode="wait">
                        {isEditMode ? (
                            <motion.div
                                key="edit"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                            >
                                <Card>
                                    <CardHeader>
                                        <CardTitle>Edit Information</CardTitle>
                                        <CardDescription>
                                            Update your personal details below.
                                        </CardDescription>
                                    </CardHeader>
                                    <CardContent className="space-y-4">
                                        <div className="grid sm:grid-cols-2 gap-4">
                                            <Input
                                                label="First Name"
                                                name="firstName"
                                                placeholder="Jhon"
                                                value={profileForm.firstName}
                                                onChange={handleProfileChange}
                                                leftIcon={<User />}
                                                rightIcon={<></>}
                                            />
                                            <Input
                                                label="Last Name"
                                                name="lastName"
                                                placeholder="Doe"
                                                value={profileForm.lastName}
                                                onChange={handleProfileChange}
                                                leftIcon={<User />}
                                                rightIcon={<></>}
                                            />
                                        </div>
                                        {/* REFACTOR: Restored the phone number field. */}
                                        <Input
                                            label="Phone Number"
                                            name="phone"
                                            value={profileForm.phone}
                                            onChange={handleProfileChange}
                                            placeholder="+91 XXXXXXXXXX"
                                            leftIcon={<Phone />}
                                            rightIcon={<></>}
                                        />
                                        <Textarea
                                            label="Bio"
                                            name="bio"
                                            value={profileForm.bio}
                                            onChange={handleProfileChange}
                                            rows={4}
                                            leftIcon={<User />}
                                            rightIcon={<></>}
                                        />
                                    </CardContent>
                                    <CardFooter className="justify-end space-x-2">
                                        <Button
                                            variant="ghost"
                                            onClick={() => setIsEditMode(false)}
                                        >
                                            Cancel
                                        </Button>
                                        <Button
                                            onClick={handleProfileSave}
                                            isLoading={
                                                updateProfileMutation.isPending
                                            }
                                        >
                                            <Save className="mr-2 h-4 w-4" />{" "}
                                            Save
                                        </Button>
                                    </CardFooter>
                                </Card>
                            </motion.div>
                        ) : isPasswordMode ? (
                            <motion.div
                                key="password"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                            >
                                <Card>
                                    <CardHeader>
                                        <CardTitle>Change Password</CardTitle>
                                        <CardDescription>
                                            Enter your current and new password.
                                        </CardDescription>
                                    </CardHeader>
                                    <CardContent className="space-y-4">
                                        <Input
                                            label="Current Password"
                                            name="currentPassword"
                                            type="password"
                                            placeholder="Current Password"
                                            value={passwordForm.currentPassword}
                                            onChange={handlePasswordChange}
                                            leftIcon={<Lock />}
                                            rightIcon={<></>}
                                        />
                                        <div className="grid sm:grid-cols-2 gap-4">
                                            <Input
                                                label="New Password"
                                                name="newPassword"
                                                type="password"
                                                placeholder="New Password"
                                                value={passwordForm.newPassword}
                                                onChange={handlePasswordChange}
                                                leftIcon={<Lock />}
                                                rightIcon={<></>}
                                            />
                                            <Input
                                                label="Confirm New Password"
                                                name="confirmPassword"
                                                type="password"
                                                placeholder="Confirm New Password"
                                                value={
                                                    passwordForm.confirmPassword
                                                }
                                                onChange={handlePasswordChange}
                                                leftIcon={<Lock />}
                                                rightIcon={<></>}
                                            />
                                        </div>
                                    </CardContent>
                                    <CardFooter className="justify-end space-x-2">
                                        <Button
                                            variant="ghost"
                                            onClick={() =>
                                                setIsPasswordMode(false)
                                            }
                                        >
                                            Cancel
                                        </Button>
                                        <Button
                                            onClick={handlePasswordSave}
                                            isLoading={
                                                updatePasswordMutation.isPending
                                            }
                                        >
                                            <Save className="mr-2 h-4 w-4" />{" "}
                                            Update Password
                                        </Button>
                                    </CardFooter>
                                </Card>
                            </motion.div>
                        ) : (
                            <motion.div
                                key="view"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                            >
                                <Card>
                                    <CardHeader>
                                        <CardTitle>
                                            Personal Information
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent className="grid sm:grid-cols-2 gap-8">
                                        <InfoField
                                            label="Full Name"
                                            value={
                                                user.firstName +
                                                " " +
                                                user.lastName
                                            }
                                            icon={User}
                                        />
                                        <InfoField
                                            label="Email Address"
                                            value={user.email}
                                            icon={Mail}
                                        />
                                        {/* REFACTOR: Restored the phone number field. */}
                                        {user.phone !== null && (
                                            <InfoField
                                                label="Phone Number"
                                                value={user.phone}
                                                icon={Phone}
                                            />
                                        )}
                                        <div className="sm:col-span-2">
                                            <label className="text-sm font-medium text-light-muted-text dark:text-dark-muted-text">
                                                Bio
                                            </label>
                                            <p className="text-base mt-1 whitespace-pre-wrap">
                                                {user.bio || "No bio provided."}
                                            </p>
                                        </div>
                                    </CardContent>
                                </Card>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </div>

            <Modal
                isOpen={isAvatarMode}
                onClose={() => setIsAvatarMode(false)}
                title="Update Avatar"
                footer={
                    <>
                        <Button
                            variant="destructive"
                            onClick={handleAvatarDelete}
                            isLoading={deleteAvatarMutation.isPending}
                            className="mr-auto"
                        >
                            <Trash2 className="mr-2 h-4 w-4" /> Remove
                        </Button>
                        <Button
                            variant="outline"
                            onClick={() => setIsAvatarMode(false)}
                        >
                            Cancel
                        </Button>
                        <Button
                            onClick={handleAvatarSave}
                            isLoading={updateAvatarMutation.isPending}
                        >
                            <Save className="mr-2 h-4 w-4" /> Save
                        </Button>
                    </>
                }
            >
                <div className="flex flex-col items-center gap-4">
                    <img
                        src={
                            avatarUrl ||
                            `https://ui-avatars.com/api/?name=${user.firstName}+${user.lastName}&background=random&color=fff`
                        }
                        alt="Avatar Preview"
                        className="w-32 h-32 rounded-full object-cover"
                    />
                    <Input
                        label="Avatar URL"
                        value={avatarUrl}
                        onChange={(e) => setAvatarUrl(e.target.value)}
                        placeholder="https://example.com/avatar.jpg"
                    />
                </div>
            </Modal>
        </div>
    );
};

export default Profile;
