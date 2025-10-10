// src/components/auth/LoginForm.jsx

import { useState } from "react";
import { useAuth } from "../../contexts/AuthContext.jsx";
import { Eye, EyeOff, Mail, Lock, LogIn, AlertCircle } from "lucide-react";
import { Button } from "../ui/Button";
import { Input } from "../ui/Input";

const LoginForm = ({ onSwitchToSignup }) => {
  const [formData, setFormData] = useState({ email: "", password: "" });
  const [showPassword, setShowPassword] = useState(false);
  const [emailError, setEmailError] = useState("");
  const { login, isLoggingIn } = useAuth();

  // Email validation regex
  const isValidEmail = (email) => {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));

    // Real-time email validation
    if (name === "email") {
      if (value && !isValidEmail(value)) {
        setEmailError("Please enter a valid email address");
      } else {
        setEmailError("");
      }
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // The `login` function from AuthContext is already wrapped in a try/catch,
    // handles API call, updates local state, shows toasts, and navigates on success.
    login(formData.email, formData.password);
    // REMOVED: If login is successful, we get redirected to the /dashboard page by AuthContext
    // REMOVED: router.push("/dashboard");
  };

  const passwordToggle = (
    <button
      type="button"
      onClick={() => setShowPassword(!showPassword)}
      aria-label="Toggle password visibility"
      className="focus:outline-none" // Add focus outline for accessibility
    >
      {showPassword ? (
        <EyeOff className="h-5 w-5" />
      ) : (
        <Eye className="h-5 w-5" />
      )}
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
        leftIcon={<Mail className="h-5 w-5" />}
        rightIcon={<></>}
        error={emailError}
      />
      <Input
        name="password"
        type={showPassword ? "text" : "password"}
        required
        value={formData.password}
        onChange={handleChange}
        placeholder="••••••••"
        disabled={isLoggingIn}
        leftIcon={<Lock className="h-5 w-5" />}
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
            disabled={isLoggingIn} // Disable button while loading
          >
            Sign Up
          </button>
        </p>
      </div>
    </form>
  );
};

export default LoginForm;
