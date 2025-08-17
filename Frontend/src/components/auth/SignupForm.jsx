// src/components/auth/SignupForm.jsx

import React, { useState } from "react";
import { useAuth } from "../../contexts/AuthContext.jsx";
import { validationToast } from "../../utils/toast.js";
import { Eye, EyeOff, Mail, Lock, User } from "lucide-react";
import { Button } from "../ui/Button";
import { Input } from "../ui/Input";

const SignupForm = ({ onSwitchToLogin }) => {
    const [formData, setFormData] = useState({
        firstName: "",
        lastName: "",
        email: "",
        password: "",
        confirmPassword: "",
    });
    const [showPassword, setShowPassword] = useState(false);
    const [showConfirmPassword, setShowConfirmPassword] = useState(false);
    const { signup, isSigningUp } = useAuth();

    const handleChange = (e) => {
        setFormData((prev) => ({ ...prev, [e.target.name]: e.target.value }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (formData.password.length < 6)
            return validationToast.minLength("Password", 6);
        if (formData.password !== formData.confirmPassword)
            return validationToast.mismatch("Passwords", "Passwords");

        try {
            await signup({
                firstName: formData.firstName,
                lastName: formData.lastName,
                email: formData.email,
                password: formData.password,
            });
            onSwitchToLogin();
        } catch (error) {
            console.error("Signup failed:", error);
        }
    };

    // REFACTOR: Interactive elements to be passed as props.
    const passwordToggle = (
        <button
            type="button"
            onClick={() => setShowPassword(!showPassword)}
            aria-label="Toggle password visibility"
        >
            {showPassword ? <EyeOff /> : <Eye />}
        </button>
    );
    const confirmPasswordToggle = (
        <button
            type="button"
            onClick={() => setShowConfirmPassword(!showConfirmPassword)}
            aria-label="Toggle confirm password visibility"
        >
            {showConfirmPassword ? <EyeOff /> : <Eye />}
        </button>
    );

    return (
        <form onSubmit={handleSubmit} className="space-y-6">
            <div>
                <h2 className="text-3xl font-bold text-light-text dark:text-dark-text">
                    Create an Account
                </h2>
                <p className="text-light-muted-text dark:text-dark-muted-text">
                    Get started by creating your new account.
                </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <Input
                    name="firstName"
                    type="text"
                    required
                    value={formData.firstName}
                    onChange={handleChange}
                    placeholder="First Name"
                    disabled={isSigningUp}
                    leftIcon={<User />}
                    rightIcon={<></>}
                />
                <Input
                    name="lastName"
                    type="text"
                    required
                    value={formData.lastName}
                    onChange={handleChange}
                    placeholder="Last Name"
                    disabled={isSigningUp}
                    leftIcon={<User />}
                    rightIcon={<></>}
                />
            </div>

            <Input
                name="email"
                type="email"
                required
                value={formData.email}
                onChange={handleChange}
                placeholder="john@example.com"
                disabled={isSigningUp}
                leftIcon={<Mail />}
                rightIcon={<></>}
            />

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {/* REFACTOR: Correctly passing the password toggles to the rightIcon prop. */}
                <Input
                    name="password"
                    type={showPassword ? "text" : "password"}
                    required
                    value={formData.password}
                    onChange={handleChange}
                    placeholder="Password"
                    disabled={isSigningUp}
                    leftIcon={<Lock />}
                    rightIcon={passwordToggle}
                />
                <Input
                    name="confirmPassword"
                    type={showConfirmPassword ? "text" : "password"}
                    required
                    value={formData.confirmPassword}
                    onChange={handleChange}
                    placeholder="Confirm Password"
                    disabled={isSigningUp}
                    leftIcon={<Lock />}
                    rightIcon={confirmPasswordToggle}
                />
            </div>

            <Button
                type="submit"
                isLoading={isSigningUp}
                className="w-full"
                size="lg"
            >
                {!isSigningUp && <User className="w-5 h-5 mr-2" />}
                {isSigningUp ? "Creating Account..." : "Create Account"}
            </Button>

            <div className="text-center pt-4">
                <p className="text-sm text-light-muted-text dark:text-dark-muted-text">
                    Already have an account?{" "}
                    <button
                        type="button"
                        onClick={onSwitchToLogin}
                        className="font-semibold text-primary-main hover:underline"
                    >
                        Sign In
                    </button>
                </p>
            </div>
        </form>
    );
};

export default SignupForm;
