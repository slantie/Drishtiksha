// src/components/auth/SignupForm.jsx

import React, { useState } from "react";
import { useAuth } from "../../hooks/useAuth.js";
import { authToast, validationToast, showToast } from "../../utils/toast.js";
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
    const [isLoading, setIsLoading] = useState(false);
    const { signup } = useAuth();

    const handleChange = (e) => {
        setFormData((prev) => ({ ...prev, [e.target.name]: e.target.value }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (formData.password.length < 6) {
            return validationToast.minLength("Password", 6);
        }
        if (formData.password !== formData.confirmPassword) {
            return validationToast.mismatch("Passwords");
        }
        setIsLoading(true);
        try {
            await signup({
                firstName: formData.firstName,
                lastName: formData.lastName,
                email: formData.email,
                password: formData.password,
            });
            authToast.signupSuccess();
            showToast.info("Account created! Please log in.");
            onSwitchToLogin();
        } catch (error) {
            authToast.signupError(error.message || "Account creation failed.");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <form onSubmit={handleSubmit} className="space-y-4">
            <div className="text-center mb-6">
                <div className="text-2xl flex items-center justify-center font-mono font-bold text-light-text dark:text-dark-text mb-2">
                    <User className="w-6 h-6 mr-3 text-light-highlight dark:text-dark-highlight" />
                    CREATE ACCOUNT
                    <div className="ml-3 w-2 h-5 bg-light-highlight dark:bg-dark-highlight animate-pulse"></div>
                </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
                <div className="group">
                    <label className="block text-sm font-mono text-light-muted-text dark:text-dark-muted-text mb-1">
                        FIRST NAME:
                    </label>
                    <input
                        name="firstName"
                        type="text"
                        required
                        value={formData.firstName}
                        onChange={handleChange}
                        className="w-full px-3 py-2.5 bg-light-muted-background/50 dark:bg-dark-muted-background/50 border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-lg text-light-text dark:text-dark-text focus:outline-none focus:border-light-highlight dark:focus:border-dark-highlight transition-all duration-300 text-sm font-mono"
                        placeholder="John"
                        disabled={isLoading}
                    />
                </div>
                <div className="group">
                    <label className="block text-sm font-mono text-light-muted-text dark:text-dark-muted-text mb-1">
                        LAST NAME:
                    </label>
                    <input
                        name="lastName"
                        type="text"
                        required
                        value={formData.lastName}
                        onChange={handleChange}
                        className="w-full px-3 py-2.5 bg-light-muted-background/50 dark:bg-dark-muted-background/50 border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-lg text-light-text dark:text-dark-text focus:outline-none focus:border-light-highlight dark:focus:border-dark-highlight transition-all duration-300 text-sm font-mono"
                        placeholder="Doe"
                        disabled={isLoading}
                    />
                </div>
            </div>
            <div className="group">
                <label className="block text-sm font-mono text-light-muted-text dark:text-dark-muted-text mb-1">
                    EMAIL ADDRESS:
                </label>
                <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <Mail className="h-4 w-4 text-light-muted-text dark:text-dark-muted-text" />
                    </div>
                    <input
                        name="email"
                        type="email"
                        required
                        value={formData.email}
                        onChange={handleChange}
                        className="w-full pl-9 pr-3 py-2.5 bg-light-muted-background/50 dark:bg-dark-muted-background/50 border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-lg text-light-text dark:text-dark-text focus:outline-none focus:border-light-highlight dark:focus:border-dark-highlight transition-all duration-300 text-sm font-mono"
                        placeholder="john@example.com"
                        disabled={isLoading}
                    />
                </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
                <div className="group">
                    <label className="block text-sm font-mono text-light-muted-text dark:text-dark-muted-text mb-1">
                        PASSWORD:
                    </label>
                    <div className="relative">
                        <input
                            name="password"
                            type={showPassword ? "text" : "password"}
                            required
                            value={formData.password}
                            onChange={handleChange}
                            className="w-full pr-10 px-3 py-2.5 bg-light-muted-background/50 dark:bg-dark-muted-background/50 border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-lg text-light-text dark:text-dark-text focus:outline-none focus:border-light-highlight dark:focus:border-dark-highlight transition-all duration-300 text-sm font-mono"
                            placeholder="••••••••"
                            disabled={isLoading}
                        />
                        <button
                            type="button"
                            onClick={() => setShowPassword(!showPassword)}
                            className="absolute inset-y-0 right-0 pr-3 flex items-center text-light-muted-text dark:text-dark-muted-text hover:text-light-highlight dark:hover:text-dark-highlight transition-colors"
                            disabled={isLoading}
                        >
                            {showPassword ? (
                                <EyeOff className="h-4 w-4" />
                            ) : (
                                <Eye className="h-4 w-4" />
                            )}
                        </button>
                    </div>
                </div>
                <div className="group">
                    <label className="block text-sm font-mono text-light-muted-text dark:text-dark-muted-text mb-1">
                        CONFIRM PASSWORD:
                    </label>
                    <div className="relative">
                        <input
                            name="confirmPassword"
                            type={showConfirmPassword ? "text" : "password"}
                            required
                            value={formData.confirmPassword}
                            onChange={handleChange}
                            className="w-full pr-10 px-3 py-2.5 bg-light-muted-background/50 dark:bg-dark-muted-background/50 border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-lg text-light-text dark:text-dark-text focus:outline-none focus:border-light-highlight dark:focus:border-dark-highlight transition-all duration-300 text-sm font-mono"
                            placeholder="••••••••"
                            disabled={isLoading}
                        />
                        <button
                            type="button"
                            onClick={() =>
                                setShowConfirmPassword(!showConfirmPassword)
                            }
                            className="absolute inset-y-0 right-0 pr-3 flex items-center text-light-muted-text dark:text-dark-muted-text hover:text-light-highlight dark:hover:text-dark-highlight transition-colors"
                            disabled={isLoading}
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
                disabled={isLoading}
                className="w-full bg-gradient-to-r from-light-highlight to-light-highlight/90 dark:from-dark-highlight dark:to-dark-highlight/90 text-white font-bold py-3 px-4 rounded-lg shadow-lg hover:shadow-xl hover:shadow-light-highlight/30 dark:hover:shadow-dark-highlight/30 transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 flex items-center justify-center space-x-2 font-mono text-sm border border-white/20"
            >
                {isLoading ? (
                    <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>CREATING ACCOUNT...</span>
                    </>
                ) : (
                    <>
                        <span>CREATE ACCOUNT</span>
                        <ArrowRight className="w-4 h-4" />
                    </>
                )}
            </button>
            <div className="text-center pt-4 border-t border-light-muted-text/10 dark:border-dark-muted-text/10">
                <p className="text-sm text-light-muted-text dark:text-dark-muted-text font-mono">
                    Already have an account?{" "}
                    <button
                        type="button"
                        onClick={onSwitchToLogin}
                        className="text-light-highlight dark:text-dark-highlight hover:underline transition-all duration-300 hover:scale-105 inline-block"
                    >
                        Login
                    </button>
                </p>
            </div>
        </form>
    );
};

export default SignupForm;
