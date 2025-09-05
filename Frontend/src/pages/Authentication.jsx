// src/pages/Authentication.jsx

import { useSearchParams } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { ShieldCheck } from "lucide-react";
import LoginForm from "../components/auth/LoginForm.jsx";
import SignupForm from "../components/auth/SignupForm.jsx";
import { config } from "../config/env.js";

function Authentication() {
  const [searchParams, setSearchParams] = useSearchParams({ view: "login" });
  const currentView = searchParams.get("view") || "login";

  const handleViewTransition = (newView) => {
    if (newView !== currentView) {
      setSearchParams({ view: newView });
    }
  };

  // Framer Motion variants for form transitions
  const formVariants = {
    initial: (direction) => ({
      opacity: 0,
      x: direction === "right" ? 50 : -50,
      transition: { duration: 0.01 }, // Shorter transition for initial to prevent flicker
    }),
    animate: {
      opacity: 1,
      x: 0,
      transition: { duration: 0.3, ease: "easeInOut" },
    },
    exit: (direction) => ({
      opacity: 0,
      x: direction === "right" ? -50 : 50,
      transition: { duration: 0.2, ease: "easeInOut" },
    }),
  };

  // Determine transition direction based on current view
  const direction = currentView === "login" ? "left" : "right";

  return (
    <div className="w-full max-w-6xl mx-auto grid lg:grid-cols-2 bg-light-background dark:bg-dark-muted-background rounded-2xl shadow-2xl overflow-hidden min-h-[70vh]">
      {" "}
      {/* Added min-h for consistent height */}
      {/* Left Panel: Brand Information (Hidden on small screens) */}
      <div className="hidden lg:flex flex-col items-center justify-center p-12 text-center bg-light-muted-background dark:bg-dark-background border-r border-light-secondary dark:border-dark-secondary space-y-6">
        {" "}
        {/* Consistent vertical spacing */}
        <img
          src="/Logo.svg"
          alt="Drishtiksha Logo"
          className="w-24 h-24 mb-6"
        />{" "}
        {/* Added alt text */}
        <h1 className="text-4xl font-bold">{config.VITE_PROJECT_NAME}</h1>
        <p className="text-lg text-light-muted-text dark:text-dark-muted-text max-w-sm mx-auto">
          {" "}
          {/* Added max-w for readability */}
          Advanced AI-powered analysis to ensure the authenticity of your
          digital media.
        </p>
        <div className="mt-8 flex items-center space-x-2 text-green-600 dark:text-green-400">
          <ShieldCheck className="w-6 h-6" /> {/* Consistent icon size */}
          <span className="text-base font-medium">
            Secure & Reliable Platform
          </span>{" "}
          {/* Consistent text size */}
        </div>
      </div>
      {/* Right Panel: Authentication Forms */}
      <div className="p-8 sm:p-12 flex flex-col justify-center min-h-[550px] lg:min-h-full overflow-hidden">
        {" "}
        {/* Adjusted min-h for responsiveness */}
        <AnimatePresence mode="wait" custom={direction}>
          <motion.div
            key={currentView}
            custom={direction}
            variants={formVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            className="w-full"
          >
            {currentView === "login" ? (
              <LoginForm
                onSwitchToSignup={() => handleViewTransition("signup")}
              />
            ) : (
              <SignupForm
                onSwitchToLogin={() => handleViewTransition("login")}
              />
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}

export default Authentication;
