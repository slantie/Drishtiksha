// src/components/auth/LoginForm.jsx

import React, { useState, useEffect } from "react";
import { useAuth } from "../../contexts/AuthContext.jsx";
import { validationToast } from "../../utils/toast.js";
import { Eye, EyeOff, Mail, Lock, LogIn } from "lucide-react";
import { Button } from "../ui/Button";
import { Input } from "../ui/Input";
import { useNavigate } from "react-router-dom";
import { useProfileQuery } from "../../hooks/useAuthQuery.js";

const LoginForm = ({ onSwitchToSignup }) => {
    const [formData, setFormData] = useState({
        email: "",
        password: "",
    });
    const [showPassword, setShowPassword] = useState(false);
    const { login, isLoggingIn } = useAuth();

    const navigate = useNavigate();
    const { data: profile, isLoading } = useProfileQuery();

    useEffect(() => {
        if (!isLoading && profile) {
            navigate("/dashboard");
        }
    }, [isLoading, profile, navigate]);

    // while profile is loading, avoid flicker
    if (isLoading) {
        return <p className="text-center">Checking session...</p>;
    }

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData((prev) => ({
            ...prev,
            [name]: value,
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!formData.email || !formData.password) {
            return validationToast.required("Email and Password");
        }
        await login(formData.email, formData.password);
    };

    // REFACTOR: The password toggle is now an interactive element passed as a prop.
    const passwordToggle = (
        <button
            type="button"
            onClick={() => setShowPassword(!showPassword)}
            aria-label="Toggle password visibility"
        >
            {showPassword ? <EyeOff /> : <Eye />}
        </button>
    );

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

            <Input
                name="email"
                type="email"
                required
                value={formData.email}
                onChange={handleChange}
                placeholder="user@domain.com"
                disabled={isLoggingIn}
                leftIcon={<Mail />}
                rightIcon={<></>}
            />

            {/* REFACTOR: Correctly passing the password toggle to the rightIcon prop. */}
            <Input
                name="password"
                type={showPassword ? "text" : "password"}
                required
                value={formData.password}
                onChange={handleChange}
                placeholder="••••••••"
                disabled={isLoggingIn}
                leftIcon={<Lock />}
                rightIcon={passwordToggle}
            />

            <Button
                type="submit"
                isLoading={isLoggingIn}
                className="w-full"
                size="lg"
            >
                {!isLoggingIn && <LogIn className="w-5 h-5 mr-2" />}
                {isLoggingIn ? "Authenticating..." : "Sign In"}
            </Button>

            <div className="text-center pt-4">
                <p className="text-sm text-light-muted-text dark:text-dark-muted-text">
                    Don't have an account?{" "}
                    <button
                        type="button"
                        onClick={onSwitchToSignup}
                        className="font-semibold text-primary-main hover:underline"
                    >
                        Sign Up
                    </button>
                </p>
            </div>
        </form>
    );
};

export default LoginForm;
