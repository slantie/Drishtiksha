// src/pages/Authentication.jsx

import React, { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { ShieldCheck } from "lucide-react";
import LoginForm from "../components/auth/LoginForm.jsx";
import SignupForm from "../components/auth/SignupForm.jsx";
import ThemeToggle from "../components/ThemeToggle.jsx";

const projectName = import.meta.env.VITE_PROJECT_NAME;

function Authentication() {
    const [searchParams] = useSearchParams();
    const [currentView, setCurrentView] = useState("login");
    const [isTransitioning, setIsTransitioning] = useState(false);

    useEffect(() => {
        // Set view based on URL query param on initial load
        const view = searchParams.get("view");
        if (view === "signup") {
            setCurrentView("signup");
        }
    }, [searchParams]);

    const handleViewTransition = (newView) => {
        if (newView === currentView) return;
        setIsTransitioning(true);
        setTimeout(() => {
            setCurrentView(newView);
            window.history.pushState(null, "", `/auth?view=${newView}`);
            setIsTransitioning(false);
        }, 200); // Short transition for a snappy feel
    };

    return (
        <div className="w-screen h-screen flex flex-col items-center justify-center bg-light-background dark:bg-dark-background">
            <div className="fixed top-5 right-5 z-50">
                <ThemeToggle />
            </div>
            <div className="flex items-center justify-center">
                <div className="mx-auto grid lg:grid-cols-2 bg-light-background dark:bg-dark-background rounded-2xl shadow-2xl overflow-hidden">
                    <div className="hidden lg:flex flex-col items-center justify-center p-12 text-center">
                        <img
                            src="/Logo.svg"
                            alt="Logo"
                            className="w-24 h-24 mb-6"
                        />
                        <h1 className="text-4xl font-bold text-light-text dark:text-dark-text">
                            {projectName}
                        </h1>
                        <p className="text-lg text-light-muted-text dark:text-dark-muted-text mt-4">
                            Advanced AI-powered analysis to ensure the
                            authenticity of your digital media.
                        </p>
                        <div className="mt-8 flex items-center space-x-2 text-green-600 dark:text-green-400">
                            <ShieldCheck className="w-5 h-5" />
                            <span className="text-sm font-medium">
                                Secure & Reliable
                            </span>
                        </div>
                    </div>

                    {/* Right Panel: Authentication Forms */}
                    <div className="p-8 sm:p-12 flex flex-col justify-center">
                        <div
                            className={`transition-opacity duration-200 ${
                                isTransitioning ? "opacity-0" : "opacity-100"
                            }`}
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
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default Authentication;
