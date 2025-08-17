// src/components/auth/SignupForm.jsx

import React, { useState } from "react";
import { useAuth } from "../../contexts/AuthContext.jsx";
import { validationToast } from "../../utils/toast.js";
import {
    Eye,
    EyeOff,
    Mail,
    Lock,
    Loader2,
    ArrowRight,
    User,
} from "lucide-react";

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
            return validationToast.mismatch("Passwords");

        try {
            await signup({
                firstName: formData.firstName,
                lastName: formData.lastName,
                email: formData.email,
                password: formData.password,
            });
            onSwitchToLogin(); // Switch to login view after successful signup
        } catch (error) {
            console.error("Signup failed:", error);
        }
    };

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

            <div className="grid grid-cols-2 gap-4">
                <div>
                    <label className="block text-sm font-medium text-light-muted-text dark:text-dark-muted-text mb-1">
                        First Name
                    </label>
                    <input
                        name="firstName"
                        type="text"
                        required
                        value={formData.firstName}
                        onChange={handleChange}
                        className="w-full px-3 py-2.5 bg-light-muted-background dark:bg-dark-muted-background border border-light-secondary dark:border-dark-secondary rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-main/50"
                        placeholder="John"
                        disabled={isSigningUp}
                    />
                </div>
                <div>
                    <label className="block text-sm font-medium text-light-muted-text dark:text-dark-muted-text mb-1">
                        Last Name
                    </label>
                    <input
                        name="lastName"
                        type="text"
                        required
                        value={formData.lastName}
                        onChange={handleChange}
                        className="w-full px-3 py-2.5 bg-light-muted-background dark:bg-dark-muted-background border border-light-secondary dark:border-dark-secondary rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-main/50"
                        placeholder="Doe"
                        disabled={isSigningUp}
                    />
                </div>
            </div>

            <div>
                <label className="block text-sm font-medium text-light-muted-text dark:text-dark-muted-text mb-1">
                    Email Address
                </label>
                <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <Mail className="h-4 w-4 text-gray-400" />
                    </div>
                    <input
                        name="email"
                        type="email"
                        required
                        value={formData.email}
                        onChange={handleChange}
                        className="w-full pl-10 pr-3 py-2.5 bg-light-muted-background dark:bg-dark-muted-background border border-light-secondary dark:border-dark-secondary rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-main/50"
                        placeholder="john@example.com"
                        disabled={isSigningUp}
                    />
                </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
                <div>
                    <label className="block text-sm font-medium text-light-muted-text dark:text-dark-muted-text mb-1">
                        Password
                    </label>
                    <div className="relative">
                        <input
                            name="password"
                            type={showPassword ? "text" : "password"}
                            required
                            value={formData.password}
                            onChange={handleChange}
                            className="w-full pr-10 px-3 py-2.5 bg-light-muted-background dark:bg-dark-muted-background border border-light-secondary dark:border-dark-secondary rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-main/50"
                            placeholder="••••••••"
                            disabled={isSigningUp}
                        />
                        <button
                            type="button"
                            onClick={() => setShowPassword(!showPassword)}
                            className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-500 hover:text-primary-main"
                        >
                            {showPassword ? (
                                <EyeOff className="h-4 w-4" />
                            ) : (
                                <Eye className="h-4 w-4" />
                            )}
                        </button>
                    </div>
                </div>
                <div>
                    <label className="block text-sm font-medium text-light-muted-text dark:text-dark-muted-text mb-1">
                        Confirm Password
                    </label>
                    <div className="relative">
                        <input
                            name="confirmPassword"
                            type={showConfirmPassword ? "text" : "password"}
                            required
                            value={formData.confirmPassword}
                            onChange={handleChange}
                            className="w-full pr-10 px-3 py-2.5 bg-light-muted-background dark:bg-dark-muted-background border border-light-secondary dark:border-dark-secondary rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-main/50"
                            placeholder="••••••••"
                            disabled={isSigningUp}
                        />
                        <button
                            type="button"
                            onClick={() =>
                                setShowConfirmPassword(!showConfirmPassword)
                            }
                            className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-500 hover:text-primary-main"
                        >
                            {showConfirmPassword ? (
                                <EyeOff className="h-4 w-4" />
                            ) : (
                                <Eye className="h-4 w-4" />
                            )}
                        </button>
                    </div>
                </div>
            </div>

            <button
                type="submit"
                disabled={isSigningUp}
                className="w-full flex items-center justify-center bg-primary-main text-white font-semibold py-3 px-4 rounded-lg shadow-md hover:bg-primary-main/90 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
            >
                {isSigningUp ? (
                    <>
                        <Loader2 className="w-5 h-5 mr-2 animate-spin" />{" "}
                        Creating Account...
                    </>
                ) : (
                    <>
                        <User className="w-5 h-5 mr-2" /> Create Account
                    </>
                )}
            </button>

            <div className="text-center pt-4">
                <p className="text-sm text-light-muted-text dark:text-dark-muted-text">
                    Already have an account?{" "}
                    <button
                        type="button"
                        onClick={onSwitchToLogin}
                        className="text-primary-main font-semibold hover:underline"
                    >
                        Sign In
                    </button>
                </p>
            </div>
        </form>
    );
};

export default SignupForm;
