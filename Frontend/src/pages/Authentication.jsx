import React, { useState, useEffect } from "react";
import {
  Eye,
  EyeOff,
  Mail,
  Lock,
  Loader2,
  Power,
  Home,
  ArrowRight,
  User,
} from "lucide-react";
import ThemeToggle from "../components/ThemeToggle";
import { authToast, validationToast, showToast } from "../utils/toast";

const projectName = import.meta.env.VITE_PROJECT_NAME;
const backendUrl = import.meta.env.VITE_BACKEND_URL;

function Authentication() {
  const [currentView, setCurrentView] = useState("login");
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [monitorPower, setMonitorPower] = useState(true);

  const [loginData, setLoginData] = useState({
    email: "",
    password: "",
    rememberMe: false,
  });

  const [signupData, setSignupData] = useState({
    firstName: "",
    lastName: "",
    email: "",
    password: "",
    confirmPassword: "",
    agreeToTerms: false,
  });

  // Check URL params for initial view
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const view = urlParams.get("view");
    if (view === "signup") {
      setCurrentView("signup");
    }
  }, []);

  const handleViewTransition = (newView) => {
    setIsTransitioning(true);
    setTimeout(() => {
      setCurrentView(newView);
      setIsTransitioning(false);
      window.history.pushState({}, "", `/auth?view=${newView}`);
    }, 300);
  };

  const handleLoginChange = (e) => {
    const { name, type, checked, value } = e.target;
    setLoginData({
      ...loginData,
      [name]: type === "checkbox" ? checked : value,
    });
  };

  const handleSignupChange = (e) => {
    const { name, type, checked, value } = e.target;
    setSignupData({
      ...signupData,
      [name]: type === "checkbox" ? checked : value,
    });
  };

  // Validation functions remain the same
  const validateLogin = () => {
    if (!loginData.email) {
      validationToast.required("Email");
      return false;
    }
    if (!/\S+@\S+\.\S+/.test(loginData.email)) {
      validationToast.invalid("email address");
      return false;
    }
    if (!loginData.password) {
      validationToast.required("Password");
      return false;
    }
    return true;
  };

  const validateSignup = () => {
    if (!signupData.firstName || !signupData.lastName) {
      validationToast.required("First and last name");
      return false;
    }
    if (!signupData.email) {
      validationToast.required("Email");
      return false;
    }
    if (!/\S+@\S+\.\S+/.test(signupData.email)) {
      validationToast.invalid("email address");
      return false;
    }
    if (signupData.password.length < 6) {
      validationToast.minLength("Password", 6);
      return false;
    }
    if (signupData.password !== signupData.confirmPassword) {
      validationToast.mismatch("Passwords", "confirm password");
      return false;
    }
    return true;
  };

  // Updated login handler
  const handleLogin = async (e) => {
    e.preventDefault();
    if (!validateLogin()) return;

    setIsLoading(true);
    const loadingToast = authToast.authenticating();

    try {
      console.log("Attempting login to:", `${backendUrl}/api/auth/login`);

      const response = await fetch(`${backendUrl}/api/auth/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          // Add any additional headers if needed
        },
        credentials: "include", // Include cookies if needed
        body: JSON.stringify({
          email: loginData.email.trim(),
          password: loginData.password,
          rememberMe: loginData.rememberMe,
        }),
      });

      console.log("Login response status:", response.status);
      const data = await response.json();
      console.log("Login response data:", data);

      showToast.dismiss(loadingToast);

      if (response.ok && data.success) {
        authToast.loginSuccess();

        // Store auth data
        const storage = loginData.rememberMe ? localStorage : sessionStorage;
        storage.setItem("authToken", data.token);
        storage.setItem("user", JSON.stringify(data.user));
        storage.setItem("isAuthenticated", "true");

        // Clear form
        setLoginData({
          email: "",
          password: "",
          rememberMe: false,
        });

        // Check for redirect URL
        const urlParams = new URLSearchParams(window.location.search);
        const redirectPath = urlParams.get("redirect");

        setTimeout(() => {
          if (redirectPath) {
            window.location.href = decodeURIComponent(redirectPath);
          } else {
            window.location.href = "/";
          }
        }, 2000);
      } else {
        // Handle specific error cases
        if (response.status === 401) {
          authToast.loginError("Invalid email or password");
        } else if (response.status === 400) {
          authToast.loginError(data.message || "Please check your input");
        } else {
          authToast.loginError(data.message || "Login failed");
        }
      }
    } catch (error) {
      showToast.dismiss(loadingToast);
      console.error("Login error:", error);

      if (error.name === "TypeError" && error.message.includes("fetch")) {
        authToast.connectionError(
          "Cannot connect to server. Please check if the backend is running."
        );
      } else {
        authToast.connectionError("Network error. Please try again.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  // Updated signup handler
  const handleSignup = async (e) => {
    e.preventDefault();
    if (!validateSignup()) return;

    setIsLoading(true);
    const loadingToast = authToast.creatingAccount();

    try {
      console.log("Attempting signup to:", `${backendUrl}/api/auth/signup`);

      const response = await fetch(`${backendUrl}/api/auth/signup`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include", // Include cookies if needed
        body: JSON.stringify({
          firstName: signupData.firstName.trim(),
          lastName: signupData.lastName.trim(),
          email: signupData.email.trim(),
          password: signupData.password,
        }),
      });

      console.log("Signup response status:", response.status);
      const data = await response.json();
      console.log("Signup response data:", data);

      showToast.dismiss(loadingToast);

      if (response.ok && data.success) {
        authToast.signupSuccess();

        // Clear form
        setSignupData({
          firstName: "",
          lastName: "",
          email: "",
          password: "",
          confirmPassword: "",
          agreeToTerms: false,
        });

        setTimeout(() => {
          // Pre-fill login email
          setLoginData((prev) => ({
            ...prev,
            email: signupData.email.trim(),
            password: "",
            rememberMe: false,
          }));
          handleViewTransition("login");
          showToast.info(
            "Account created! Please log in with your credentials"
          );
        }, 2000);
      } else {
        // Handle specific error cases
        if (response.status === 409) {
          authToast.signupError("An account with this email already exists");
        } else if (response.status === 400) {
          authToast.signupError(data.message || "Please check your input");
        } else {
          authToast.signupError(data.message || "Account creation failed");
        }
      }
    } catch (error) {
      showToast.dismiss(loadingToast);
      console.error("Signup error:", error);

      if (error.name === "TypeError" && error.message.includes("fetch")) {
        authToast.connectionError(
          "Cannot connect to server. Please check if the backend is running."
        );
      } else {
        authToast.connectionError("Network error. Please try again.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const toggleMonitorPower = () => {
    setMonitorPower(!monitorPower);
    showToast.info(monitorPower ? "Monitor turned off" : "Monitor turned on", {
      duration: 2000,
      icon: monitorPower ? "📺" : "💻",
    });
  };

  const goHome = () => {
    window.location.href = "/";
  };

  return (
    <div className="flex items-center justify-center py-2">
      <div className="relative">
        {/* Monitor Frame */}
        <div className="relative bg-gradient-to-b from-gray-800 via-gray-850 to-gray-900 p-4 rounded-3xl shadow-2xl border border-gray-700">
          <div className="relative">
            {/* Screen Bezel */}
            <div className="bg-gradient-to-b from-gray-900 to-black p-4 rounded-2xl shadow-inner">
              <div className="flex flex-col">
                {/* Main Screen */}
                <div
                  className={`w-[72rem] h-[40rem] rounded-xl transition-all duration-500 relative overflow-hidden ${
                    monitorPower
                      ? "bg-gradient-to-br from-light-background to-light-muted-background dark:from-dark-background dark:to-dark-muted-background"
                      : "bg-black"
                  }`}
                >
                  {/* Screen reflection effect */}
                  {monitorPower && (
                    <div className="absolute inset-0 bg-gradient-to-br from-white/5 via-transparent to-transparent pointer-events-none rounded-xl"></div>
                  )}

                  {monitorPower ? (
                    <div className="p-8 overflow-y-auto relative z-10">
                      {/* Terminal Header */}
                      <div className="flex items-center justify-between mb-6 pb-4 border-b border-light-muted-text/20 dark:border-dark-muted-text/20">
                        <div className="flex items-center space-x-2">
                          <div className="w-3 h-3 rounded-full bg-red-400 shadow-lg shadow-red-400/30"></div>
                          <div className="w-3 h-3 rounded-full bg-yellow-400 shadow-lg shadow-yellow-400/30"></div>
                          <div className="w-3 h-3 rounded-full bg-green-400 shadow-lg shadow-green-400/30 animate-pulse"></div>
                        </div>
                        <div className="text-sm text-light-muted-text dark:text-dark-muted-text font-mono flex items-center space-x-2">
                          <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                          <span>{projectName} Authentication v2.1.0</span>
                        </div>
                        <div className="flex items-center justify-center space-x-4">
                          <button
                            onClick={goHome}
                            className="bg-light-muted-background dark:bg-dark-muted-background p-2 rounded-lg border-2 border-light-muted-text/20 hover:border-light-highlight dark:hover:border-dark-highlight dark:border-dark-muted-text/20"
                          >
                            <Home className="w-5 h-5" />
                          </button>
                          <div>
                            <ThemeToggle className="w-4 h-4" />
                          </div>
                        </div>
                      </div>

                      {/* Content Container with Transition */}
                      <div
                        className={`transition-all duration-300 ${
                          isTransitioning
                            ? "opacity-0 scale-95"
                            : "opacity-100 scale-100"
                        }`}
                      >
                        {/* View Header */}
                        <div className="text-center mb-6">
                          <div className="text-2xl flex items-center justify-center font-mono font-bold text-light-text dark:text-dark-text mb-2">
                            {currentView === "login" ? (
                              <>
                                <Lock className="w-6 h-6 mr-3 text-light-highlight dark:text-dark-highlight" />
                                USER LOGIN
                              </>
                            ) : (
                              <>
                                <User className="w-6 h-6 mr-3 text-light-highlight dark:text-dark-highlight" />
                                CREATE ACCOUNT
                              </>
                            )}
                            <div className="ml-3 w-2 h-5 bg-light-highlight dark:bg-dark-highlight animate-pulse"></div>
                          </div>
                        </div>

                        {currentView === "login" ? (
                          <LoginForm
                            formData={loginData}
                            onChange={handleLoginChange}
                            onSubmit={handleLogin}
                            showPassword={showPassword}
                            setShowPassword={setShowPassword}
                            isLoading={isLoading}
                            onSwitchToSignup={() =>
                              handleViewTransition("signup")
                            }
                          />
                        ) : (
                          <SignupForm
                            formData={signupData}
                            onChange={handleSignupChange}
                            onSubmit={handleSignup}
                            showPassword={showPassword}
                            setShowPassword={setShowPassword}
                            showConfirmPassword={showConfirmPassword}
                            setShowConfirmPassword={setShowConfirmPassword}
                            isLoading={isLoading}
                            onSwitchToLogin={() =>
                              handleViewTransition("login")
                            }
                          />
                        )}

                        {/* Environment Info */}
                        {import.meta.env.NODE_ENV === "development" && (
                          <div className="mt-4 text-sm text-light-muted-text dark:text-dark-muted-text bg-light-muted-background/30 dark:bg-dark-muted-background/30 p-2 rounded font-mono border border-light-muted-text/10 dark:border-dark-muted-text/10">
                            <span className="text-yellow-400">⚠</span> DEV MODE
                            - Backend: {backendUrl}
                          </div>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="h-full flex items-center justify-center">
                      <div className="text-center">
                        <div className="w-16 h-16 mx-auto mb-4 bg-gray-800 rounded-full flex items-center justify-center shadow-lg">
                          <Power className="w-8 h-8 text-gray-600" />
                        </div>
                        <p className="text-gray-600 text-sm font-mono">
                          Monitor is off
                        </p>
                        <p className="text-gray-700 text-sm mt-1 font-mono">
                          Press power button to turn on
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* LED indicator */}
            <div className="absolute bottom-2 right-2">
              <div
                className={`w-2 h-2 rounded-full transition-all duration-300 ${
                  monitorPower
                    ? "bg-green-400 shadow-lg shadow-green-400/50 animate-pulse"
                    : "bg-gray-600"
                }`}
              ></div>
            </div>
          </div>

          {/* Monitor Controls */}
          <div className="text-center mx-2 mt-4 flex items-center justify-between">
            <div>
              <button
                onClick={toggleMonitorPower}
                className={`w-8 h-8 rounded-full border-2 flex items-center justify-center transition-all duration-300 transform hover:scale-110 ${
                  monitorPower
                    ? "bg-green-400 border-green-500 shadow-lg shadow-green-400/30"
                    : "bg-gray-600 border-gray-700 hover:bg-gray-500"
                }`}
                title="Power"
              >
                <Power className="w-4 h-4 text-white" />
              </button>
            </div>
            <div className="text-gray-400 text-sm font-bold">
              Slantie Industries
            </div>
          </div>
        </div>

        {/* Ambient lighting effect */}
        {monitorPower && (
          <div className="absolute inset-0 -z-10 bg-gradient-to-r from-light-highlight/5 to-transparent dark:from-dark-highlight/5 rounded-3xl blur-xl"></div>
        )}
      </div>
    </div>
  );
}

// Login Form Component (remains the same)
function LoginForm({
  formData,
  onChange,
  onSubmit,
  showPassword,
  setShowPassword,
  isLoading,
  onSwitchToSignup,
}) {
  return (
    <form onSubmit={onSubmit} className="space-y-4">
      {/* Email Field */}
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
            onChange={onChange}
            className="w-full pl-9 pr-3 py-2.5 bg-light-muted-background/50 dark:bg-dark-muted-background/50 border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-lg text-light-text dark:text-dark-text focus:outline-none focus:border-light-highlight dark:focus:border-dark-highlight focus:shadow-lg focus:shadow-light-highlight/20 dark:focus:shadow-dark-highlight/20 transition-all duration-300 text-sm font-mono hover:border-light-highlight/50 dark:hover:border-dark-highlight/50"
            placeholder="user@domain.com"
            disabled={isLoading}
          />
        </div>
      </div>

      {/* Password Field */}
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
            onChange={onChange}
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

      {/* Remember Me & Forgot Password */}
      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center">
          <input
            id="rememberMe"
            name="rememberMe"
            type="checkbox"
            checked={formData.rememberMe}
            onChange={onChange}
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

      {/* Submit Button */}
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

      {/* Switch to Signup */}
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
}

// Signup Form Component (remains the same)
function SignupForm({
  formData,
  onChange,
  onSubmit,
  showPassword,
  setShowPassword,
  showConfirmPassword,
  setShowConfirmPassword,
  isLoading,
  onSwitchToLogin,
}) {
  return (
    <form onSubmit={onSubmit} className="space-y-4">
      {/* Name Fields */}
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
            onChange={onChange}
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
            onChange={onChange}
            className="w-full px-3 py-2.5 bg-light-muted-background/50 dark:bg-dark-muted-background/50 border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-lg text-light-text dark:text-dark-text focus:outline-none focus:border-light-highlight dark:focus:border-dark-highlight transition-all duration-300 text-sm font-mono"
            placeholder="Doe"
            disabled={isLoading}
          />
        </div>
      </div>

      {/* Email Field */}
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
            onChange={onChange}
            className="w-full pl-9 pr-3 py-2.5 bg-light-muted-background/50 dark:bg-dark-muted-background/50 border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-lg text-light-text dark:text-dark-text focus:outline-none focus:border-light-highlight dark:focus:border-dark-highlight transition-all duration-300 text-sm font-mono"
            placeholder="john@example.com"
            disabled={isLoading}
          />
        </div>
      </div>

      {/* Password Fields */}
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
              onChange={onChange}
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
              onChange={onChange}
              className="w-full pr-10 px-3 py-2.5 bg-light-muted-background/50 dark:bg-dark-muted-background/50 border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-lg text-light-text dark:text-dark-text focus:outline-none focus:border-light-highlight dark:focus:border-dark-highlight transition-all duration-300 text-sm font-mono"
              placeholder="••••••••"
              disabled={isLoading}
            />
            <button
              type="button"
              onClick={() => setShowConfirmPassword(!showConfirmPassword)}
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

      {/* Submit Button */}
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

      {/* Switch to Login */}
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
}

export default Authentication;
