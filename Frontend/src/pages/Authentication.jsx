// src/pages/Authentication.jsx

import React, { useState, useEffect } from "react";
import { Power, Home } from "lucide-react";
import ThemeToggle from "../components/ThemeToggle";
import { showToast } from "../utils/toast";
import LoginForm from "../components/auth/LoginForm.jsx";
import SignupForm from "../components/auth/SignupForm.jsx";

const projectName = import.meta.env.VITE_PROJECT_NAME;

function Authentication() {
    const [currentView, setCurrentView] = useState("login");
    const [isTransitioning, setIsTransitioning] = useState(false);
    const [monitorPower, setMonitorPower] = useState(true);

    useEffect(() => {
        const params = new URLSearchParams(window.location.search);
        if (params.get("view") === "signup") {
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

    const toggleMonitorPower = () => {
        setMonitorPower(!monitorPower);
        showToast.info(
            monitorPower ? "Monitor turned off" : "Monitor turned on",
            {
                duration: 2000,
                icon: monitorPower ? "📺" : "💻",
            }
        );
    };

    const goHome = () => {
        window.location.href = "/";
    };

    return (
        <div className="flex items-center justify-center py-2">
            <div className="relative">
                <div className="relative bg-gradient-to-b from-gray-800 via-gray-850 to-gray-900 p-4 rounded-3xl shadow-2xl border border-gray-700">
                    <div className="relative">
                        <div className="bg-gradient-to-b from-gray-900 to-black p-4 rounded-2xl shadow-inner">
                            <div
                                className={`w-[72rem] h-[40rem] rounded-xl transition-all duration-500 relative overflow-hidden ${
                                    monitorPower
                                        ? "bg-gradient-to-br from-light-background to-light-muted-background dark:from-dark-background dark:to-dark-muted-background"
                                        : "bg-black"
                                }`}
                            >
                                {monitorPower && (
                                    <div className="absolute inset-0 bg-gradient-to-br from-white/5 via-transparent to-transparent pointer-events-none rounded-xl"></div>
                                )}
                                {monitorPower ? (
                                    <div className="p-8 overflow-y-auto relative z-10">
                                        <div className="flex items-center justify-between mb-6 pb-4 border-b border-light-muted-text/20 dark:border-dark-muted-text/20">
                                            <div className="flex items-center space-x-2">
                                                <div className="w-3 h-3 rounded-full bg-red-400 shadow-lg shadow-red-400/30"></div>
                                                <div className="w-3 h-3 rounded-full bg-yellow-400 shadow-lg shadow-yellow-400/30"></div>
                                                <div className="w-3 h-3 rounded-full bg-green-400 shadow-lg shadow-green-400/30 animate-pulse"></div>
                                            </div>
                                            <div className="text-sm text-light-muted-text dark:text-dark-muted-text font-mono flex items-center space-x-2">
                                                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                                                <span>
                                                    {projectName} Authentication
                                                    v2.1.0
                                                </span>
                                            </div>
                                            <div className="flex items-center justify-center space-x-4">
                                                <button
                                                    onClick={goHome}
                                                    className="bg-light-muted-background dark:bg-dark-muted-background p-2 rounded-lg border-2 border-light-muted-text/20 hover:border-light-highlight dark:hover:border-dark-highlight dark:border-dark-muted-text/20"
                                                >
                                                    <Home className="w-5 h-5" />
                                                </button>
                                                <ThemeToggle />
                                            </div>
                                        </div>
                                        <div
                                            className={`transition-all duration-300 ${
                                                isTransitioning
                                                    ? "opacity-0 scale-95"
                                                    : "opacity-100 scale-100"
                                            }`}
                                        >
                                            {currentView === "login" ? (
                                                <LoginForm
                                                    onSwitchToSignup={() =>
                                                        handleViewTransition(
                                                            "signup"
                                                        )
                                                    }
                                                />
                                            ) : (
                                                <SignupForm
                                                    onSwitchToLogin={() =>
                                                        handleViewTransition(
                                                            "login"
                                                        )
                                                    }
                                                />
                                            )}
                                        </div>
                                    </div>
                                ) : (
                                    <div className="h-full flex items-center justify-center text-center">
                                        <div>
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
                    <div className="text-center mx-2 mt-4 flex items-center justify-between">
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
                        <div className="text-gray-400 text-sm font-bold">
                            Slantie Industries
                        </div>
                    </div>
                </div>
                {monitorPower && (
                    <div className="absolute inset-0 -z-10 bg-gradient-to-r from-light-highlight/5 to-transparent dark:from-dark-highlight/5 rounded-3xl blur-xl"></div>
                )}
            </div>
        </div>
    );
}

export default Authentication;
