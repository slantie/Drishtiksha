// src/pages/Authentication.jsx

import { useSearchParams } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { ShieldCheck } from "lucide-react";
import LoginForm from "../components/auth/LoginForm.jsx";
import SignupForm from "../components/auth/SignupForm.jsx";

const projectName = import.meta.env.VITE_PROJECT_NAME;

function Authentication() {
    const [searchParams, setSearchParams] = useSearchParams({ view: "login" });
    const currentView = searchParams.get("view") || "login";

    const handleViewTransition = (newView) => {
        if (newView !== currentView) {
            setSearchParams({ view: newView });
        }
    };

    const formVariants = {
        initial: (direction) => ({
            opacity: 0,
            x: direction === "right" ? 50 : -50,
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

    const direction = currentView === "login" ? "left" : "right";

    // REFACTOR: The entire two-panel card layout is now self-contained within this component for precise control.
    return (
        <div className="w-full max-w-[80vw] mx-auto grid lg:grid-cols-2 bg-light-background dark:bg-dark-muted-background rounded-2xl shadow-2xl overflow-hidden">
            {/* Left Panel: Brand Information */}
            <div className="hidden lg:flex flex-col items-center justify-center p-12 text-center bg-light-muted-background dark:bg-dark-background border-r border-light-secondary dark:border-dark-secondary">
                <img src="/Logo.svg" alt="Logo" className="w-24 h-24 mb-6" />
                <h1 className="text-4xl font-bold text-light-text dark:text-dark-text">
                    {projectName}
                </h1>
                <p className="text-lg text-light-muted-text dark:text-dark-muted-text mt-4">
                    Advanced AI-powered analysis to ensure the authenticity of
                    your digital media.
                </p>
                <div className="mt-8 flex items-center space-x-2 text-green-600 dark:text-green-400">
                    <ShieldCheck className="w-5 h-5" />
                    <span className="text-sm font-medium">
                        Secure & Reliable
                    </span>
                </div>
            </div>

            {/* Right Panel: Authentication Forms */}
            {/* REFACTOR: This container now correctly centers the form content vertically and provides proper padding. */}
            <div className="p-8 sm:p-12 flex flex-col justify-center min-h-[550px] overflow-hidden">
                <AnimatePresence mode="wait" custom={direction}>
                    <motion.div
                        key={currentView}
                        custom={direction}
                        variants={formVariants}
                        initial="initial"
                        animate="animate"
                        exit="exit"
                    >
                        {currentView === "login" ? (
                            <LoginForm
                                onSwitchToSignup={() =>
                                    handleViewTransition("signup")
                                }
                            />
                        ) : (
                            <SignupForm
                                onSwitchToLogin={() =>
                                    handleViewTransition("login")
                                }
                            />
                        )}
                    </motion.div>
                </AnimatePresence>
            </div>
        </div>
    );
}

export default Authentication;
