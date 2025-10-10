// src/components/auth/SignupForm.jsx

import { useState } from "react";
import { useAuth } from "../../contexts/AuthContext.jsx";
import { Eye, EyeOff, Mail, Lock, User, UserPlus, CheckCircle, XCircle } from "lucide-react";
import { Button } from "../ui/Button";
import { Input } from "../ui/Input";
import { showToast } from "../../utils/toast.jsx";

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
  const [emailError, setEmailError] = useState("");
  const [passwordError, setPasswordError] = useState("");
  const { signup, isSigningUp } = useAuth();

  // Email validation
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

    // Real-time password validation
    if (name === "password") {
      if (value && value.length < 6) {
        setPasswordError("Password must be at least 6 characters");
      } else {
        setPasswordError("");
      }
    }

    // Real-time confirm password validation
    if (name === "confirmPassword") {
      if (value && value !== formData.password) {
        setPasswordError("Passwords do not match");
      } else if (formData.password.length < 6) {
        setPasswordError("Password must be at least 6 characters");
      } else {
        setPasswordError("");
      }
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (formData.password !== formData.confirmPassword) {
      showToast.error("Passwords do not match.");
      return;
    }

    try {
      await signup({
        firstName: formData.firstName,
        lastName: formData.lastName,
        email: formData.email,
        password: formData.password,
      });
      // On success, AuthContext handles the success toast and navigation to /auth?view=login.
      // We just need to trigger the parent to switch views if it's not already handled by AuthContext's navigation.
      // If AuthContext navigates, this call might be redundant, but harmless.
      onSwitchToLogin();
    } catch (error) {
      // Errors are already handled and toasted by the useSignupMutation hook.
      // We can log here for extra debugging if needed.
      console.error("Signup failed component-level:", error);
    }
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
  const confirmPasswordToggle = (
    <button
      type="button"
      onClick={() => setShowConfirmPassword(!showConfirmPassword)}
      aria-label="Toggle confirm password visibility"
      className="focus:outline-none" // Add focus outline for accessibility
    >
      {showConfirmPassword ? (
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
          leftIcon={<User className="h-5 w-5" />}
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
          leftIcon={<User className="h-5 w-5" />}
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
        leftIcon={<Mail className="h-5 w-5" />}
        rightIcon={<></>}
        error={emailError}
      />

      {/* Password Requirements */}
      {formData.password && (
        <div className="space-y-2 p-3 bg-light-background dark:bg-dark-muted-background rounded-lg">
          <p className="text-xs font-medium text-light-muted-text dark:text-dark-muted-text">
            Password Requirements:
          </p>
          <div className="flex items-center gap-2 text-xs">
            {formData.password.length >= 6 ? (
              <CheckCircle className="h-4 w-4 text-green-500" />
            ) : (
              <XCircle className="h-4 w-4 text-red-500" />
            )}
            <span className={formData.password.length >= 6 ? "text-green-600 dark:text-green-400" : "text-light-muted-text dark:text-dark-muted-text"}>
              At least 6 characters
            </span>
          </div>
          {formData.confirmPassword && (
            <div className="flex items-center gap-2 text-xs">
              {formData.password === formData.confirmPassword ? (
                <CheckCircle className="h-4 w-4 text-green-500" />
              ) : (
                <XCircle className="h-4 w-4 text-red-500" />
              )}
              <span className={formData.password === formData.confirmPassword ? "text-green-600 dark:text-green-400" : "text-light-muted-text dark:text-dark-muted-text"}>
                Passwords match
              </span>
            </div>
          )}
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <Input
          name="password"
          type={showPassword ? "text" : "password"}
          required
          value={formData.password}
          onChange={handleChange}
          placeholder="Password"
          disabled={isSigningUp}
          leftIcon={<Lock className="h-5 w-5" />}
          rightIcon={passwordToggle}
          minLength={6}
          error={passwordError && !formData.confirmPassword ? passwordError : ""}
        />
        <Input
          name="confirmPassword"
          type={showConfirmPassword ? "text" : "password"}
          required
          value={formData.confirmPassword}
          onChange={handleChange}
          placeholder="Confirm Password"
          disabled={isSigningUp}
          leftIcon={<Lock className="h-5 w-5" />}
          rightIcon={confirmPasswordToggle}
          minLength={6}
          error={formData.confirmPassword && passwordError ? passwordError : ""}
        />
      </div>

      <Button
        type="submit"
        isLoading={isSigningUp}
        className="w-full"
        size="lg"
        disabled={isSigningUp || formData.password.length < 6 || formData.password !== formData.confirmPassword} // Disable if passwords don't match
      >
        {!isSigningUp && <UserPlus className="w-5 h-5 mr-2" />}
        {isSigningUp ? "Creating Account..." : "Create Account"}
      </Button>

      <div className="text-center pt-4">
        <p className="text-sm text-light-muted-text dark:text-dark-muted-text">
          Already have an account?{" "}
          <button
            type="button"
            onClick={onSwitchToLogin}
            className="font-semibold text-primary-main hover:underline"
            disabled={isSigningUp}
          >
            Sign In
          </button>
        </p>
      </div>
    </form>
  );
};

export default SignupForm;
