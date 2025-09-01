// src/components/auth/LoginForm.jsx

import { useState } from "react";
import { useAuth } from "../../contexts/AuthContext.jsx";
import { Eye, EyeOff, Mail, Lock, LogIn } from "lucide-react";
import { Button } from "../ui/Button";
import { Input } from "../ui/Input";

const LoginForm = ({ onSwitchToSignup }) => {
    const [formData, setFormData] = useState({ email: "", password: "" });
    const [showPassword, setShowPassword] = useState(false);
    const { login, isLoggingIn } = useAuth();

    // REMOVED: The redirect logic is now correctly handled by the <PublicRoute> component.
    // This simplifies the form's responsibility to just handling login.

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData((prev) => ({ ...prev, [name]: value }));
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        // The `login` function from AuthContext is already wrapped in a try/catch
        // and will show a toast on failure, so we don't need to add it here.
        login(formData.email, formData.password);
    };

    const passwordToggle = (
        <button type="button" onClick={() => setShowPassword(!showPassword)} aria-label="Toggle password visibility">
            {showPassword ? <EyeOff /> : <Eye />}
        </button>
    );

    return (
        <form onSubmit={handleSubmit} className="space-y-6">
            <div>
                <h2 className="text-3xl font-bold text-light-text dark:text-dark-text">Welcome Back</h2>
                <p className="text-light-muted-text dark:text-dark-muted-text">Please sign in to continue.</p>
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
            />
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
            <Button type="submit" isLoading={isLoggingIn} className="w-full" size="lg">
                {!isLoggingIn && <LogIn className="w-5 h-5 mr-2" />}
                {isLoggingIn ? "Authenticating..." : "Sign In"}
            </Button>
            <div className="text-center pt-4">
                <p className="text-sm text-light-muted-text dark:text-dark-muted-text">
                    Don't have an account?{" "}
                    <button type="button" onClick={onSwitchToSignup} className="font-semibold text-primary-main hover:underline">
                        Sign Up
                    </button>
                </p>
            </div>
        </form>
    );
};

export default LoginForm;