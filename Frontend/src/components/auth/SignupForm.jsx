// src/components/auth/SignupForm.jsx

import { useState } from "react";
import { useAuth } from "../../contexts/AuthContext.jsx";
import { Eye, EyeOff, Mail, Lock, User } from "lucide-react";
import { Button } from "../ui/Button";
import { Input } from "../ui/Input";

const SignupForm = ({ onSwitchToLogin }) => {
  const [formData, setFormData] = useState({
    firstName: "",
    lastName: "",
    email: "",
    password: "",
    confirmPassword: "", // Kept for UI, but validation is on the backend
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const { signup, isSigningUp } = useAuth();

  const handleChange = (e) => {
    setFormData((prev) => ({ ...prev, [e.target.name]: e.target.value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    // REMOVED: Client-side validation is now handled robustly by the backend with Zod.
    // This simplifies the component and avoids duplicating validation logic.
    try {
      await signup({
        firstName: formData.firstName,
        lastName: formData.lastName,
        email: formData.email,
        password: formData.password,
      });
      // On success, the useSignupMutation hook shows a toast and we switch views.
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
