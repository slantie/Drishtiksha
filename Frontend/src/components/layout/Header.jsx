// src/components/layout/Header.jsx

import React, { useState, useEffect } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { User, LogOut, Menu, X, ChevronDown } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import ThemeToggle from "../ThemeToggle";
import { useAuth } from "../../contexts/AuthContext.jsx";
import { Button } from "../ui/Button";

const projectName = import.meta.env.VITE_PROJECT_NAME || "Drishtiksha";

const navItems = [
    
];

const authenticatedNavItems = [
    { path: "/dashboard", label: "Dashboard" },
    { path: "/monitor", label: "System Monitoring" },
];

function Header() {
    // REFACTOR: All state and hooks are preserved.
    const { isAuthenticated, user, logout } = useAuth();
    const [showUserMenu, setShowUserMenu] = useState(false);
    const [showMobileMenu, setShowMobileMenu] = useState(false);
    const [scrolled, setScrolled] = useState(false);
    const navigate = useNavigate();

    // REFACTOR: Added a scroll effect for a more dynamic header.
    useEffect(() => {
        const handleScroll = () => setScrolled(window.scrollY > 10);
        window.addEventListener("scroll", handleScroll);
        return () => window.removeEventListener("scroll", handleScroll);
    }, []);

    useEffect(() => {
        const handleClickOutside = (event) => {
            if (showUserMenu && !event.target.closest(".user-menu-container"))
                setShowUserMenu(false);
            if (
                showMobileMenu &&
                !event.target.closest(".mobile-menu-container")
            )
                setShowMobileMenu(false);
        };
        document.addEventListener("mousedown", handleClickOutside);
        return () =>
            document.removeEventListener("mousedown", handleClickOutside);
    }, [showUserMenu, showMobileMenu]);

    const handleProfileClick = () => {
        setShowUserMenu(false);
        setShowMobileMenu(false);
        navigate("/profile");
    };

    const dropdownVariants = {
        hidden: { opacity: 0, scale: 0.95, y: -10 },
        visible: {
            opacity: 1,
            scale: 1,
            y: 0,
            transition: { duration: 0.2, ease: "easeOut" },
        },
        exit: {
            opacity: 0,
            scale: 0.95,
            y: -10,
            transition: { duration: 0.15 },
        },
    };

    return (
        <header
            className={`sticky top-0 z-50 transition-all duration-300 border-b-2 border-light-noisy-background/40 dark:border-dark-muted-background ${
                scrolled
                    ? "border-b border-light-secondary dark:border-dark-secondary bg-light-background/80 dark:bg-dark-background/80 backdrop-blur-xl"
                    : ""
            }`}
        >
            <div className="mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between items-center h-20">
                    {/* Logo */}
                    <div
                        className="flex items-center space-x-3 cursor-pointer"
                        onClick={() => navigate("/")}
                    >
                        <img src="/Logo.svg" alt="Logo" className="w-10 h-10" />
                        <span className="text-xl font-bold tracking-tight">
                            {projectName}
                        </span>
                    </div>

                    {/* Desktop Navigation */}
                    <nav className="hidden lg:flex items-center space-x-8">
                        {(isAuthenticated
                            ? authenticatedNavItems
                            : navItems
                        ).map((item) => (
                            <NavLink
                                key={item.path}
                                to={item.path}
                                className={({ isActive }) =>
                                    `text-md font-semibold transition-colors ${
                                        isActive
                                            ? "text-primary-main"
                                            : "hover:text-primary-main"
                                    }`
                                }
                            >
                                {item.label}
                            </NavLink>
                        ))}
                    </nav>

                    {/* Actions */}
                    <div className="hidden lg:flex items-center space-x-3">
                        <ThemeToggle />
                        {isAuthenticated && user ? (
                            <div className="relative user-menu-container">
                                <button
                                    onClick={() =>
                                        setShowUserMenu(!showUserMenu)
                                    }
                                    className="flex items-center space-x-2"
                                >
                                    <div className="w-10 h-10 bg-primary-main/10 rounded-full flex items-center justify-center text-primary-main font-bold">
                                        {user.avatar ? (
                                            <img
                                                src={user.avatar}
                                                alt="User Avatar"
                                                className="w-full h-full rounded-full object-cover"
                                            />
                                        ) : (
                                            user.firstName
                                                ?.charAt(0)
                                                .toUpperCase()
                                        )}
                                    </div>
                                </button>
                                <AnimatePresence>
                                    {showUserMenu && (
                                        <motion.div
                                            variants={dropdownVariants}
                                            initial="hidden"
                                            animate="visible"
                                            exit="exit"
                                            className="absolute right-0 mt-2 w-64 bg-light-background dark:bg-dark-muted-background rounded-lg shadow-lg border border-light-secondary dark:border-dark-secondary overflow-hidden"
                                        >
                                            <div className="p-4 border-b border-light-secondary dark:border-dark-secondary">
                                                <p className="font-semibold">
                                                    {user.firstName}{" "}
                                                    {user.lastName}
                                                </p>
                                                <p className="text-sm text-light-muted-text dark:text-dark-muted-text">
                                                    {user.email}
                                                </p>
                                            </div>
                                            <div className="p-2">
                                                <button
                                                    onClick={handleProfileClick}
                                                    className="w-full text-left flex items-center gap-2 p-2 rounded-md hover:bg-light-hover dark:hover:bg-dark-hover"
                                                >
                                                    {" "}
                                                    <User className="h-4 w-4" />{" "}
                                                    Profile{" "}
                                                </button>
                                                <button
                                                    onClick={logout}
                                                    className="w-full text-left flex items-center gap-2 p-2 rounded-md hover:bg-light-hover dark:hover:bg-dark-hover text-red-500"
                                                >
                                                    {" "}
                                                    <LogOut className="h-4 w-4" />{" "}
                                                    Sign Out{" "}
                                                </button>
                                            </div>
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </div>
                        ) : (
                            <div className="flex items-center gap-2">
                                <Button
                                    variant="outline"
                                    onClick={() => navigate("/auth?view=login")}
                                >
                                    Login
                                </Button>
                                <Button
                                    onClick={() =>
                                        navigate("/auth?view=signup")
                                    }
                                >
                                    Sign Up
                                </Button>
                            </div>
                        )}
                    </div>

                    {/* Mobile Menu Button */}
                    <div className="lg:hidden flex items-center gap-2">
                        <ThemeToggle />
                        <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => setShowMobileMenu(true)}
                            className="mobile-menu-container"
                        >
                            <Menu />
                        </Button>
                    </div>
                </div>
            </div>
            {/* Mobile Menu Panel */}
            <AnimatePresence>
                {showMobileMenu && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="lg:hidden fixed inset-0 bg-black/50 z-50"
                        onClick={() => setShowMobileMenu(false)}
                    >
                        <motion.div
                            initial={{ x: "100%" }}
                            animate={{ x: 0 }}
                            exit={{ x: "100%" }}
                            transition={{
                                type: "spring",
                                stiffness: 300,
                                damping: 30,
                            }}
                            className="absolute top-0 right-0 h-full w-80 bg-light-background dark:bg-dark-background p-6"
                            onClick={(e) => e.stopPropagation()}
                        >
                            <div className="flex justify-between items-center mb-8">
                                <span className="text-lg font-bold">
                                    {projectName}
                                </span>
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    onClick={() => setShowMobileMenu(false)}
                                >
                                    <X />
                                </Button>
                            </div>
                            <nav className="flex flex-col space-y-4">
                                {(isAuthenticated
                                    ? authenticatedNavItems
                                    : navItems
                                ).map((item) => (
                                    <NavLink
                                        key={item.path}
                                        to={item.path}
                                        onClick={() => setShowMobileMenu(false)}
                                        className={({ isActive }) =>
                                            `text-lg font-semibold p-2 rounded-md transition-colors ${
                                                isActive
                                                    ? "bg-light-hover dark:bg-dark-hover text-primary-main"
                                                    : "hover:bg-light-hover dark:hover:bg-dark-hover"
                                            }`
                                        }
                                    >
                                        {item.label}
                                    </NavLink>
                                ))}
                            </nav>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </header>
    );
}

export default Header;
