// src/components/auth/LoginForm.jsx

import React, { useState } from "react";
import { useAuth } from "../../contexts/AuthContext.jsx";
import { validationToast } from "../../utils/toast.js";
import { Eye, EyeOff, Mail, Lock, Loader2, ArrowRight } from "lucide-react";

const LoginForm = ({ onSwitchToSignup }) => {
    const [formData, setFormData] = useState({
        email: "",
        password: "",
        rememberMe: false,
    });
    const [showPassword, setShowPassword] = useState(false);
    const { login, isLoggingIn } = useAuth();

    const handleChange = (e) => {
        const { name, type, checked, value } = e.target;
        setFormData((prev) => ({
            ...prev,
            [name]: type === "checkbox" ? checked : value,
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!formData.email || !formData.password) {
            return validationToast.required("Email and Password");
        }
        await login(formData.email, formData.password, formData.rememberMe);
    };

    return (
        <form onSubmit={handleSubmit} className="space-y-6">
            <div>
                <h2 className="text-3xl font-bold text-light-text dark:text-dark-text">
                    Welcome Back
                </h2>
                <p className="text-light-muted-text dark:text-dark-muted-text">
                    Please sign in to continue.
                </p>
            </div>

            <div className="group">
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
                        placeholder="user@domain.com"
                        disabled={isLoggingIn}
                    />
                </div>
            </div>

            <div className="group">
                <label className="block text-sm font-medium text-light-muted-text dark:text-dark-muted-text mb-1">
                    Password
                </label>
                <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <Lock className="h-4 w-4 text-gray-400" />
                    </div>
                    <input
                        name="password"
                        type={showPassword ? "text" : "password"}
                        required
                        value={formData.password}
                        onChange={handleChange}
                        className="w-full pl-10 pr-10 py-2.5 bg-light-muted-background dark:bg-dark-muted-background border border-light-secondary dark:border-dark-secondary rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-main/50"
                        placeholder="••••••••"
                        disabled={isLoggingIn}
                    />
                    <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-500 hover:text-primary-main"
                        disabled={isLoggingIn}
                    >
                        {showPassword ? (
                            <EyeOff className="h-4 w-4" />
                        ) : (
                            <Eye className="h-4 w-4" />
                        )}
                    </button>
                </div>
            </div>

            <div className="flex items-center justify-between text-sm">
                <div className="flex items-center">
                    <input
                        id="rememberMe"
                        name="rememberMe"
                        type="checkbox"
                        checked={formData.rememberMe}
                        onChange={handleChange}
                        className="h-4 w-4 rounded text-primary-main focus:ring-primary-main/50"
                        disabled={isLoggingIn}
                    />
                    <label
                        htmlFor="rememberMe"
                        className="ml-2 text-light-muted-text dark:text-dark-muted-text"
                    >
                        Remember me
                    </label>
                </div>
                <a
                    href="/forgot-password"
                    className="text-primary-main hover:underline"
                >
                    Forgot password?
                </a>
            </div>

            <button
                type="submit"
                disabled={isLoggingIn}
                className="w-full flex items-center justify-center bg-primary-main text-white font-semibold py-3 px-4 rounded-lg shadow-md hover:bg-primary-main/90 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
            >
                {isLoggingIn ? (
                    <>
                        <Loader2 className="w-5 h-5 mr-2 animate-spin" />{" "}
                        Authenticating...
                    </>
                ) : (
                    <>
                        <ArrowRight className="w-5 h-5 mr-2" /> Sign In
                    </>
                )}
            </button>

            <div className="text-center pt-4">
                <p className="text-sm text-light-muted-text dark:text-dark-muted-text">
                    Don't have an account?{" "}
                    <button
                        type="button"
                        onClick={onSwitchToSignup}
                        className="text-primary-main font-semibold hover:underline"
                    >
                        Sign Up
                    </button>
                </p>
            </div>
        </form>
    );
};

export default LoginForm;
