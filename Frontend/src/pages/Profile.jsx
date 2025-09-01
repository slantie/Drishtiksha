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

// Simplified AvatarCard: It no longer has an edit button.
const AvatarCard = ({ user }) => (
  <Card className="text-center">
    <CardContent className="p-8">
      <div className="relative inline-block">
        <div className="w-32 h-32 rounded-full border-4 border-light-secondary dark:border-dark-secondary shadow-lg bg-primary-main/20 flex items-center justify-center">
          <span className="text-5xl font-bold text-primary-main">
            {user.firstName?.charAt(0)}
            {user.lastName?.charAt(0)}
          </span>
        </div>
      </div>
      <h2 className="text-2xl font-bold mt-4">
        {user.firstName} {user.lastName}
      </h2>
      <p className="text-light-muted-text dark:text-dark-muted-text">
        {user.email}
      </p>
      <div className="mt-3 inline-flex items-center gap-2 px-3 py-1 bg-primary-main/10 text-primary-main rounded-full text-sm font-semibold capitalize">
        <Shield className="w-4 h-4" />
        {user.role?.toLowerCase() || "User"}
      </div>
    </CardContent>
  </Card>
);

const InfoField = ({ label, value, icon: Icon }) => (
  <div>
    <label className="text-sm font-medium text-light-muted-text dark:text-dark-muted-text">
      {label}
    </label>
    <div className="flex items-center gap-3 mt-1">
      <Icon className="h-5 w-5 text-primary-main flex-shrink-0" />
      <p className="text-base font-semibold truncate">
        {value || "Not provided"}
      </p>
    </div>
  </div>
);

const Profile = () => {
  const { data: user, isLoading: isProfileLoading } = useProfileQuery();
  const updateProfileMutation = useUpdateProfileMutation();
  const updatePasswordMutation = useUpdatePasswordMutation();

  // REFACTOR: Simplified state management to a single state variable.
  const [activeView, setActiveView] = useState("view"); // 'view', 'editProfile', 'editPassword'

  const [profileForm, setProfileForm] = useState({
    firstName: "",
    lastName: "",
  });
  const [passwordForm, setPasswordForm] = useState({
    currentPassword: "",
    newPassword: "",
  });

  useEffect(() => {
    if (user) {
      setProfileForm({
        firstName: user.firstName || "",
        lastName: user.lastName || "",
      });
    }
  }, [user]);

  const handleProfileChange = (e) =>
    setProfileForm({ ...profileForm, [e.target.name]: e.target.value });
  const handlePasswordChange = (e) =>
    setPasswordForm({ ...passwordForm, [e.target.name]: e.target.value });

  const handleProfileSave = () => {
    updateProfileMutation.mutate(profileForm, {
      onSuccess: () => setActiveView("view"),
    });
  };

  const handlePasswordSave = () => {
    updatePasswordMutation.mutate(passwordForm, {
      onSuccess: () => {
        setActiveView("view");
        setPasswordForm({ currentPassword: "", newPassword: "" });
      },
    });
    
  };

  if (isProfileLoading) return <PageLoader text="Loading Profile..." />;
  if (!user)
    return (
      <div className="text-center py-16">
        <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
        <h2 className="text-2xl font-bold">Profile Not Found</h2>
      </div>
    );

  return (
    <div className="space-y-8">
      <PageHeader
        title="Profile & Settings"
        description="Manage your personal information and account security."
        actions={
          activeView === "view" && (
            <div className="flex gap-2">
              <Button
                variant="outline"
                onClick={() => setActiveView("editPassword")}
              >
                <Lock className="mr-2 h-4 w-4" /> Change Password
              </Button>
              <Button onClick={() => setActiveView("editProfile")}>
                <Edit3 className="mr-2 h-4 w-4" /> Edit Profile
              </Button>
            </div>
          )
        }
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-1 space-y-8">
          <AvatarCard user={user} />
          <Card>
            <CardHeader>
              <CardTitle>Account Info</CardTitle>
            </CardHeader>
            <CardContent>
              <InfoField
                label="Member Since"
                value={new Date(user.createdAt).toLocaleDateString()}
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
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <Card>
                  <CardHeader>
                    <CardTitle>Personal Information</CardTitle>
                  </CardHeader>
                  <CardContent className="grid sm:grid-cols-2 gap-8">
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
                        value={profileForm.firstName}
                        onChange={handleProfileChange}
                        leftIcon={<User />}
                      />
                      <Input
                        label="Last Name"
                        name="lastName"
                        value={profileForm.lastName}
                        onChange={handleProfileChange}
                        leftIcon={<User />}
                      />
                    </div>
                  </CardContent>
                  <CardFooter className="justify-end space-x-2">
                    <Button
                      variant="ghost"
                      onClick={() => setActiveView("view")}
                    >
                      Cancel
                    </Button>
                    <Button
                      onClick={handleProfileSave}
                      isLoading={updateProfileMutation.isPending}
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
                      value={passwordForm.currentPassword}
                      onChange={handlePasswordChange}
                      leftIcon={<Lock />}
                    />
                    <Input
                      label="New Password"
                      name="newPassword"
                      type="password"
                      value={passwordForm.newPassword}
                      onChange={handlePasswordChange}
                      leftIcon={<Lock />}
                    />
                  </CardContent>
                  <CardFooter className="justify-end space-x-2">
                    <Button
                      variant="ghost"
                      onClick={() => setActiveView("view")}
                    >
                      Cancel
                    </Button>
                    <Button
                      onClick={handlePasswordSave}
                      isLoading={updatePasswordMutation.isPending}
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
