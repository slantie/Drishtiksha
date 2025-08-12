// src/components/auth/LoginForm.jsx

import React, { useState } from "react";
import { useAuth } from "../../hooks/useAuth.js";
import { authToast, validationToast } from "../../utils/toast.js";
import { Eye, EyeOff, Mail, Lock, Loader2, ArrowRight } from "lucide-react";

const LoginForm = ({ onSwitchToSignup }) => {
    const [formData, setFormData] = useState({
        email: "",
        password: "",
        rememberMe: false,
    });
    const [showPassword, setShowPassword] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const { login } = useAuth();

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
        setIsLoading(true);
        try {
            // The login function now handles navigation internally via useEffect.
            // We just need to wait for it to complete.
            await login(formData.email, formData.password, formData.rememberMe);
            authToast.loginSuccess();
        } catch (error) {
            authToast.loginError(error.message || "Invalid credentials.");
            setIsLoading(false); // Ensure loading is stopped on error
        }
        // No need for a finally block to set isLoading, as successful login will navigate away.
    };

    return (
        <form onSubmit={handleSubmit} className="space-y-4">
            <div className="text-center mb-6">
                <div className="text-2xl flex items-center justify-center font-mono font-bold text-light-text dark:text-dark-text mb-2">
                    <Lock className="w-6 h-6 mr-3 text-light-highlight dark:text-dark-highlight" />
                    USER LOGIN
                    <div className="ml-3 w-2 h-5 bg-light-highlight dark:bg-dark-highlight animate-pulse"></div>
                </div>
            </div>
            <div className="group">
                <label className="block text-sm font-mono text-light-muted-text dark:text-dark-muted-text mb-1 group-focus-within:text-light-highlight dark:group-focus-within:text-dark-highlight transition-colors">
                    USERNAME/EMAIL:
                </label>
                <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <Mail className="h-4 w-4 text-light-muted-text dark:text-dark-muted-text group-focus-within:text-light-highlight dark:group-focus-within:text-dark-highlight transition-colors" />
                    </div>
                    <input
                        name="email"
                        type="email"
                        required
                        value={formData.email}
                        onChange={handleChange}
                        className="w-full pl-9 pr-3 py-2.5 bg-light-muted-background/50 dark:bg-dark-muted-background/50 border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-lg text-light-text dark:text-dark-text focus:outline-none focus:border-light-highlight dark:focus:border-dark-highlight focus:shadow-lg focus:shadow-light-highlight/20 dark:focus:shadow-dark-highlight/20 transition-all duration-300 text-sm font-mono hover:border-light-highlight/50 dark:hover:border-dark-highlight/50"
                        placeholder="user@domain.com"
                        disabled={isLoading}
                    />
                </div>
            </div>
            <div className="group">
                <label className="block text-sm font-mono text-light-muted-text dark:text-dark-muted-text mb-1 group-focus-within:text-light-highlight dark:group-focus-within:text-dark-highlight transition-colors">
                    PASSWORD:
                </label>
                <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <Lock className="h-4 w-4 text-light-muted-text dark:text-dark-muted-text group-focus-within:text-light-highlight dark:group-focus-within:text-dark-highlight transition-colors" />
                    </div>
                    <input
                        name="password"
                        type={showPassword ? "text" : "password"}
                        required
                        value={formData.password}
                        onChange={handleChange}
                        className="w-full pl-9 pr-10 py-2.5 bg-light-muted-background/50 dark:bg-dark-muted-background/50 border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-lg text-light-text dark:text-dark-text focus:outline-none focus:border-light-highlight dark:focus:border-dark-highlight focus:shadow-lg focus:shadow-light-highlight/20 dark:focus:shadow-dark-highlight/20 transition-all duration-300 text-sm font-mono hover:border-light-highlight/50 dark:hover:border-dark-highlight/50"
                        placeholder="••••••••"
                        disabled={isLoading}
                    />
                    <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute inset-y-0 right-0 pr-3 flex items-center text-light-muted-text dark:text-dark-muted-text hover:text-light-highlight dark:hover:text-dark-highlight transition-all duration-300 hover:scale-110"
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
            <div className="flex items-center justify-between text-sm">
                <div className="flex items-center">
                    <input
                        id="rememberMe"
                        name="rememberMe"
                        type="checkbox"
                        checked={formData.rememberMe}
                        onChange={handleChange}
                        className="h-3 w-3 text-light-highlight dark:text-dark-highlight"
                        disabled={isLoading}
                    />
                    <label
                        htmlFor="rememberMe"
                        className="ml-2 text-light-muted-text dark:text-dark-muted-text font-mono"
                    >
                        Remember me
                    </label>
                </div>
                <a
                    href="/forgot-password"
                    className="text-light-highlight dark:text-dark-highlight hover:underline font-mono transition-all duration-300 hover:scale-105"
                >
                    Reset pwd
                </a>
            </div>
            <button
                type="submit"
                disabled={isLoading}
                className="w-full bg-gradient-to-r from-light-highlight to-light-highlight/90 dark:from-dark-highlight dark:to-dark-highlight/90 text-white font-bold py-3 px-4 rounded-lg shadow-lg hover:shadow-xl hover:shadow-light-highlight/30 dark:hover:shadow-dark-highlight/30 transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 flex items-center justify-center space-x-2 font-mono text-sm border border-white/20"
            >
                {isLoading ? (
                    <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>AUTHENTICATING...</span>
                    </>
                ) : (
                    <>
                        <span>LOGIN</span>
                        <ArrowRight className="w-4 h-4" />
                    </>
                )}
            </button>
            <div className="text-center pt-4 border-t border-light-muted-text/10 dark:border-dark-muted-text/10">
                <p className="text-sm text-light-muted-text dark:text-dark-muted-text font-mono">
                    New user?{" "}
                    <button
                        type="button"
                        onClick={onSwitchToSignup}
                        className="text-light-highlight dark:text-dark-highlight hover:underline transition-all duration-300 hover:scale-105 inline-block"
                    >
                        Create account
                    </button>
                </p>
            </div>
        </form>
    );
};

export default LoginForm;
