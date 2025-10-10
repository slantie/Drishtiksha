// src/pages/Profile.jsx

import React, { useState, useEffect } from "react";
import {
  User,
  Mail,
  Save,
  Lock,
  Calendar,
  Shield,
  Edit3,
  AlertCircle,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { PageLoader } from "../components/ui/LoadingSpinner";
import { showToast } from "../utils/toast.jsx";
import {
  useProfileQuery,
  useUpdateProfileMutation,
  useUpdatePasswordMutation,
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
import { PageHeader } from "../components/layout/PageHeader.jsx";
import { formatDate } from "../utils/formatters.js"; // Import formatDate

// Simplified AvatarCard
const AvatarCard = ({ user }) => (
  <Card className="text-center">
    <CardContent className="p-8 space-y-4">
      {" "}
      {/* Added consistent spacing */}
      <div className="relative inline-block">
        <div className="w-32 h-32 rounded-full border-4 border-light-secondary dark:border-dark-secondary shadow-lg bg-primary-main/20 flex items-center justify-center">
          <span className="text-5xl font-bold text-primary-main">
            {user.firstName?.charAt(0) || ""} {/* Ensure fallback for charAt */}
            {user.lastName?.charAt(0) || ""}
          </span>
        </div>
      </div>
      <div className="space-y-1">
        {" "}
        {/* Group name and email */}
        <h2 className="text-2xl font-bold">
          {user.firstName} {user.lastName}
        </h2>
        <p className="text-light-muted-text dark:text-dark-muted-text">
          {user.email}
        </p>
      </div>
      <div className="inline-flex items-center gap-2 px-3 py-1 bg-primary-main/10 text-primary-main rounded-full text-sm font-semibold capitalize">
        <Shield className="w-4 h-4" /> {/* Consistent icon size */}
        {user.role?.toLowerCase() || "User"}
      </div>
    </CardContent>
  </Card>
);

const InfoField = ({ label, value, icon: Icon }) => (
  <div>
    <label className="block text-sm font-medium text-light-muted-text dark:text-dark-muted-text mb-1">
      {" "}
      {/* Explicit block label */}
      {label}
    </label>
    <div className="flex items-center gap-3">
      {" "}
      {/* Consistent spacing */}
      <Icon className="h-5 w-5 text-primary-main flex-shrink-0" />{" "}
      {/* Consistent icon size */}
      <p className="text-base font-semibold truncate">
        {value || "Not provided"}
      </p>
    </div>
  </div>
);

const Profile = () => {
  const {
    data: user,
    isLoading: isProfileLoading,
    refetch,
  } = useProfileQuery(); // Added refetch
  const updateProfileMutation = useUpdateProfileMutation();
  const updatePasswordMutation = useUpdatePasswordMutation();

  const [activeView, setActiveView] = useState("view"); // 'view', 'editProfile', 'editPassword'

  const [profileForm, setProfileForm] = useState({
    firstName: "",
    lastName: "",
  });
  const [passwordForm, setPasswordForm] = useState({
    currentPassword: "",
    newPassword: "",
    confirmPassword: "",
  });

  useEffect(() => {
    if (user) {
      setProfileForm({
        firstName: user.firstName || "",
        lastName: user.lastName || "",
      });
    } else {
      // Clear forms if user becomes null (e.g., after logout or session expiry)
      setProfileForm({ firstName: "", lastName: "" });
      setPasswordForm({ currentPassword: "", newPassword: "", confirmPassword: "" });
    }
  }, [user]);

  const handleProfileChange = (e) =>
    setProfileForm({ ...profileForm, [e.target.name]: e.target.value });
  const handlePasswordChange = (e) =>
    setPasswordForm({ ...passwordForm, [e.target.name]: e.target.value });

  const handleProfileSave = () => {
    updateProfileMutation.mutate(profileForm, {
      onSuccess: () => {
        setActiveView("view");
        refetch(); // Refetch profile to show immediate updates
      },
    });
  };

  const handlePasswordSave = () => {
    // Basic client-side validation for new password length
    if (passwordForm.newPassword.length < 6) {
      showToast.error("New password must be at least 6 characters long.");
      return;
    }
    // Check if passwords match
    if (passwordForm.newPassword !== passwordForm.confirmPassword) {
      showToast.error("New passwords do not match.");
      return;
    }
    updatePasswordMutation.mutate(passwordForm, {
      onSuccess: () => {
        setActiveView("view");
        setPasswordForm({ currentPassword: "", newPassword: "", confirmPassword: "" }); // Clear password fields
      },
    });
  };

  if (isProfileLoading) return <PageLoader text="Loading Profile..." />;
  if (!user)
    return (
      <div className="text-center py-16 w-full max-w-full mx-auto">
        {" "}
        {/* Added full width styling */}
        <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
        <h2 className="text-2xl font-bold">Profile Not Found</h2>
        <p className="text-light-muted-text dark:text-dark-muted-text mt-2">
          There was an issue loading your profile. Please try logging in again.
        </p>
      </div>
    );

  // Framer Motion variants for section transitions
  const sectionVariants = {
    initial: { opacity: 0, y: 20 },
    animate: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.3, ease: "easeOut" },
    },
    exit: { opacity: 0, y: -20, transition: { duration: 0.2, ease: "easeIn" } },
  };

  return (
    <div className="space-y-6 w-full max-w-full mx-auto">
      {" "}
      {/* Consistent vertical spacing, full width */}
      <PageHeader
        title="Profile & Settings"
        description="Manage your personal information and account security."
        actions={
          activeView === "view" && (
            <div className="flex flex-col sm:flex-row gap-2">
              {" "}
              {/* Responsive button grouping */}
              <Button
                variant="outline"
                onClick={() => setActiveView("editPassword")}
                aria-label="Change password"
              >
                <Lock className="mr-2 h-4 w-4" /> Change Password
              </Button>
              <Button
                onClick={() => setActiveView("editProfile")}
                aria-label="Edit profile"
              >
                <Edit3 className="mr-2 h-4 w-4" /> Edit Profile
              </Button>
            </div>
          )
        }
      />
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {" "}
        {/* Consistent gap */}
        <div className="lg:col-span-1 space-y-6">
          {" "}
          {/* Consistent vertical spacing */}
          <AvatarCard user={user} />
          <Card>
            <CardHeader>
              <CardTitle>Account Info</CardTitle>
            </CardHeader>
            <CardContent>
              <InfoField
                label="Member Since"
                value={formatDate(user.createdAt)}
                icon={Calendar}
              />
            </CardContent>
          </Card>
        </div>
        <div className="lg:col-span-2">
          <AnimatePresence mode="wait">
            {activeView === "view" && (
              <motion.div
                key="view"
                variants={sectionVariants}
                initial="initial"
                animate="animate"
                exit="exit"
                className="w-full" // Ensure motion.div takes full width
              >
                <Card>
                  <CardHeader>
                    <CardTitle>Personal Information</CardTitle>
                  </CardHeader>
                  <CardContent className="grid sm:grid-cols-2 gap-6">
                    {" "}
                    {/* Consistent gap */}
                    <InfoField
                      label="Full Name"
                      value={`${user.firstName} ${user.lastName}`}
                      icon={User}
                    />
                    <InfoField
                      label="Email Address"
                      value={user.email}
                      icon={Mail}
                    />
                  </CardContent>
                </Card>
              </motion.div>
            )}
            {activeView === "editProfile" && (
              <motion.div
                key="edit"
                variants={sectionVariants}
                initial="initial"
                animate="animate"
                exit="exit"
                className="w-full"
              >
                <Card>
                  <CardHeader>
                    <CardTitle>Edit Information</CardTitle>
                    <CardDescription>
                      Update your personal details below.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {" "}
                    {/* Consistent vertical spacing */}
                    <div className="grid sm:grid-cols-2 gap-4">
                      {" "}
                      {/* Consistent gap */}
                      <Input
                        label="First Name"
                        name="firstName"
                        value={profileForm.firstName}
                        onChange={handleProfileChange}
                        leftIcon={<User className="h-5 w-5" />} // Consistent icon size
                        rightIcon={<></>}
                        disabled={updateProfileMutation.isPending} // Disable input while saving
                      />
                      <Input
                        label="Last Name"
                        name="lastName"
                        value={profileForm.lastName}
                        onChange={handleProfileChange}
                        leftIcon={<User className="h-5 w-5" />} // Consistent icon size
                        rightIcon={<></>}
                        disabled={updateProfileMutation.isPending} // Disable input while saving
                      />
                    </div>
                  </CardContent>
                  <CardFooter className="justify-end gap-2">
                    {" "}
                    {/* Consistent gap */}
                    <Button
                      variant="ghost"
                      onClick={() => setActiveView("view")}
                      disabled={updateProfileMutation.isPending}
                    >
                      Cancel
                    </Button>
                    <Button
                      onClick={handleProfileSave}
                      isLoading={updateProfileMutation.isPending}
                      disabled={!profileForm.firstName || !profileForm.lastName} // Disable save if fields are empty
                    >
                      <Save className="mr-2 h-4 w-4" /> Save
                    </Button>
                  </CardFooter>
                </Card>
              </motion.div>
            )}
            {activeView === "editPassword" && (
              <motion.div
                key="password"
                variants={sectionVariants}
                initial="initial"
                animate="animate"
                exit="exit"
                className="w-full"
              >
                <Card>
                  <CardHeader>
                    <CardTitle>Change Password</CardTitle>
                    <CardDescription>
                      Enter your current and new password. Minimum 6 characters.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {/* {" "} */}
                    {/* Consistent vertical spacing */}
                    <Input
                      label="Current Password"
                      name="currentPassword"
                      type="password"
                      placeholder="••••••••"
                      value={passwordForm.currentPassword}
                      onChange={handlePasswordChange}
                      leftIcon={<Lock/>} // Consistent icon size
                      rightIcon={<></>}
                      disabled={updatePasswordMutation.isPending}
                    />
                    <Input
                      label="New Password"
                      name="newPassword"
                      type="password"
                      placeholder="••••••••"
                      value={passwordForm.newPassword}
                      onChange={handlePasswordChange}
                      leftIcon={<Lock/>} // Consistent icon size
                      rightIcon={<></>}
                      minLength={6} // Client-side hint
                      disabled={updatePasswordMutation.isPending}
                    />
                    <Input
                      label="Confirm New Password"
                      name="confirmPassword"
                      type="password"
                      placeholder="••••••••"
                      value={passwordForm.confirmPassword}
                      onChange={handlePasswordChange}
                      leftIcon={<Lock/>}
                      rightIcon={<></>}
                      minLength={6}
                      disabled={updatePasswordMutation.isPending}
                      error={
                        passwordForm.confirmPassword &&
                        passwordForm.newPassword !== passwordForm.confirmPassword
                          ? "Passwords do not match"
                          : undefined
                      }
                    />
                  </CardContent>
                  <CardFooter className="justify-end gap-2">
                    {" "}
                    {/* Consistent gap */}
                    <Button
                      variant="ghost"
                      onClick={() => setActiveView("view")}
                      disabled={updatePasswordMutation.isPending}
                    >
                      Cancel
                    </Button>
                    <Button
                      onClick={handlePasswordSave}
                      isLoading={updatePasswordMutation.isPending}
                      disabled={
                        !passwordForm.currentPassword ||
                        !passwordForm.newPassword ||
                        !passwordForm.confirmPassword ||
                        passwordForm.newPassword.length < 6 ||
                        passwordForm.newPassword !== passwordForm.confirmPassword // Disable if passwords don't match
                      }
                    >
                      <Save className="mr-2 h-4 w-4" /> Update Password
                    </Button>
                  </CardFooter>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
};

export default Profile;
