// src/components/layout/Header.jsx

import React, { useState, useEffect } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { User, LogOut, Menu, X, ChevronDown } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import ThemeToggle from "../ThemeToggle";
import { useAuth } from "../../hooks/useAuth.js";

const projectName = import.meta.env.VITE_PROJECT_NAME;

const navItems = [{ path: "/", label: "Home" }];

const authenticatedNavItems = [
    { path: "/dashboard", label: "Dashboard" },
    { path: "/profile", label: "Profile" },
];

function Header() {
    // Get auth state and functions from the context
    const { isAuthenticated, user, logout } = useAuth();

    const [showUserMenu, setShowUserMenu] = useState(false);
    const [showMobileMenu, setShowMobileMenu] = useState(false);
    const navigate = useNavigate();

    // Simplified logout handler
    const handleLogout = () => {
        logout();
        setShowUserMenu(false);
        setShowMobileMenu(false);
    };

    const handleProfileClick = () => {
        setShowUserMenu(false);
        setShowMobileMenu(false);
        navigate("/profile");
    };

    const handleNavClick = () => {
        setShowMobileMenu(false);
    };

    useEffect(() => {
        const handleClickOutside = (event) => {
            if (showUserMenu && !event.target.closest(".user-menu-container")) {
                setShowUserMenu(false);
            }
            if (
                showMobileMenu &&
                !event.target.closest(".mobile-menu-container")
            ) {
                setShowMobileMenu(false);
            }
        };
        document.addEventListener("mousedown", handleClickOutside);
        return () =>
            document.removeEventListener("mousedown", handleClickOutside);
    }, [showUserMenu, showMobileMenu]);

    useEffect(() => {
        document.body.style.overflow = showMobileMenu ? "hidden" : "unset";
        return () => {
            document.body.style.overflow = "unset";
        };
    }, [showMobileMenu]);

    // Animation variants
    const headerVariants = {
        hidden: { opacity: 0, y: -20 },
        visible: {
            opacity: 1,
            y: 0,
            transition: { duration: 0.6, ease: "easeOut" },
        },
    };

    const logoVariants = {
        hover: {
            scale: 1.05,
            transition: { duration: 0.2, ease: "easeInOut" },
        },
        tap: { scale: 0.95 },
    };

    const navItemVariants = {
        hover: {
            scale: 1.05,
            transition: { duration: 0.2, ease: "easeInOut" },
        },
        tap: { scale: 0.95 },
    };

    const dropdownVariants = {
        hidden: {
            opacity: 0,
            scale: 0.95,
            y: -10,
            transition: { duration: 0.2 },
        },
        visible: {
            opacity: 1,
            scale: 1,
            y: 0,
            transition: { duration: 0.2, ease: "easeOut" },
        },
    };

    const mobileMenuVariants = {
        hidden: {
            x: "100%",
            transition: { duration: 0.3, ease: "easeInOut" },
        },
        visible: {
            x: 0,
            transition: { duration: 0.3, ease: "easeInOut" },
        },
    };

    const overlayVariants = {
        hidden: { opacity: 0 },
        visible: { opacity: 1 },
    };

    const menuItemVariants = {
        hidden: { opacity: 0, x: 20 },
        visible: (index) => ({
            opacity: 1,
            x: 0,
            transition: { delay: index * 0.1, duration: 0.3 },
        }),
    };

    return (
        <>
            <motion.header
                className="relative p-4 bg-light-background/80 dark:bg-dark-background/80 backdrop-blur-xl border-b border-light-muted-text/10 dark:border-dark-muted-text/10 shadow-sm z-50"
                variants={headerVariants}
                initial="hidden"
                animate="visible"
            >
                <div className="flex justify-between items-center">
                    <motion.div
                        className="flex items-center space-x-4 cursor-pointer group"
                        onClick={() => navigate("/")}
                        variants={logoVariants}
                        whileHover="hover"
                        whileTap="tap"
                    >
                        <motion.img
                            src="/Logo.svg"
                            alt="Logo"
                            className="w-10 h-10 drop-shadow-sm"
                            // whileHover={{ rotate: 5 }}
                            transition={{ duration: 0.2 }}
                        />
                        <motion.h1
                            className="text-xl md:text-2xl font-bold tracking-wide hover:text-light-highlight dark:hover:text-dark-highlight truncate"
                            whileHover={{ scale: 1.02 }}
                            transition={{ duration: 0.2 }}
                        >
                            {projectName}
                        </motion.h1>
                    </motion.div>

                    {/* Desktop Navigation */}
                    <div className="hidden lg:flex items-center space-x-3">
                        <nav>
                            <ul className="flex space-x-3">
                                {(isAuthenticated
                                    ? authenticatedNavItems
                                    : navItems
                                ).map((item, index) => (
                                    <motion.li
                                        key={item.path}
                                        variants={navItemVariants}
                                        whileHover="hover"
                                        whileTap="tap"
                                        custom={index}
                                        initial="hidden"
                                        animate="visible"
                                    >
                                        <NavLink
                                            to={item.path}
                                            className={({ isActive }) =>
                                                `relative px-4 py-2 text-sm font-medium tracking-wide transition-all duration-300 rounded-full ${
                                                    isActive
                                                        ? "text-white bg-gradient-to-r from-light-highlight to-light-highlight/80 dark:from-dark-highlight dark:to-dark-highlight/80 shadow-lg"
                                                        : "text-light-muted-text dark:text-dark-muted-text hover:text-light-highlight dark:hover:text-dark-highlight hover:bg-light-muted-background/50 dark:hover:bg-dark-muted-background/50"
                                                }`
                                            }
                                        >
                                            {item.label}
                                        </NavLink>
                                    </motion.li>
                                ))}
                            </ul>
                        </nav>

                        <motion.div
                            className="h-8 w-px bg-gradient-to-b from-transparent via-light-muted-text/30 to-transparent dark:text-dark-text"
                            initial={{ opacity: 0, scaleY: 0 }}
                            animate={{ opacity: 1, scaleY: 1 }}
                            transition={{ delay: 0.3, duration: 0.4 }}
                        />

                        <div className="flex items-center space-x-4">
                            <ThemeToggle />
                            {isAuthenticated && user ? (
                                <div className="relative user-menu-container z-[9999]">
                                    <motion.button
                                        onClick={() =>
                                            setShowUserMenu(!showUserMenu)
                                        }
                                        className="flex items-center space-x-3 bg-light-muted-background/50 dark:bg-dark-muted-background/50 border border-light-muted-text/20 hover:border-light-highlight dark:hover:border-dark-highlight dark:border-dark-muted-text/20 backdrop-blur-sm font-medium py-2 px-3 rounded-full transition-all duration-300 hover:shadow-lg group"
                                        whileHover={{ scale: 1.02 }}
                                        whileTap={{ scale: 0.98 }}
                                    >
                                        {user.avatar ? (
                                            <motion.img
                                                src={user.avatar}
                                                alt="Avatar"
                                                className="w-8 h-8 rounded-full object-cover border-2 border-white/20 shadow-sm"
                                                whileHover={{ scale: 1.1 }}
                                                transition={{ duration: 0.2 }}
                                            />
                                        ) : (
                                            <motion.div
                                                className="w-7 h-7 bg-gradient-to-br from-light-highlight to-light-highlight/80 dark:from-dark-highlight dark:to-dark-highlight/80 rounded-full flex items-center justify-center text-white text-sm font-bold shadow-sm"
                                                whileHover={{ scale: 1.1 }}
                                                transition={{ duration: 0.2 }}
                                            >
                                                {user.firstName
                                                    ?.charAt(0)
                                                    ?.toUpperCase() || "U"}
                                            </motion.div>
                                        )}
                                        <span className="hidden xl:inline text-sm">
                                            {user.firstName} {user.lastName}
                                        </span>
                                        <motion.div
                                            animate={{
                                                rotate: showUserMenu ? 180 : 0,
                                            }}
                                            transition={{ duration: 0.2 }}
                                        >
                                            <ChevronDown className="w-4 h-4 opacity-70 group-hover:opacity-100 transition-opacity" />
                                        </motion.div>
                                    </motion.button>

                                    <AnimatePresence>
                                        {showUserMenu && (
                                            <motion.div
                                                className="absolute right-0 mt-2 w-72 bg-light-background dark:bg-dark-background backdrop-blur-xl border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-2xl shadow-2xl overflow-hidden"
                                                variants={dropdownVariants}
                                                initial="hidden"
                                                animate="visible"
                                                exit="hidden"
                                                style={{
                                                    zIndex: 99999,
                                                }}
                                            >
                                                <div className="px-5 py-4 border-b border-light-muted-text/10 dark:border-dark-muted-text/10 bg-gradient-to-r from-light-muted-background/30 to-transparent dark:from-dark-muted-background/30">
                                                    <div className="flex items-center space-x-4">
                                                        {user.avatar ? (
                                                            <img
                                                                src={
                                                                    user.avatar
                                                                }
                                                                alt="Avatar"
                                                                className="w-14 h-14 rounded-full object-cover border-2 border-white/20 shadow-sm"
                                                            />
                                                        ) : (
                                                            <div className="w-12 h-12 bg-gradient-to-br from-light-highlight to-light-highlight/80 dark:from-dark-highlight dark:to-dark-highlight/80 rounded-full flex items-center justify-center text-white font-bold shadow-sm">
                                                                {user.firstName
                                                                    ?.charAt(0)
                                                                    ?.toUpperCase() ||
                                                                    "U"}
                                                            </div>
                                                        )}
                                                        <div>
                                                            <div className="font-semibold text-light-text dark:text-dark-text">
                                                                {user.firstName}{" "}
                                                                {user.lastName}
                                                            </div>
                                                            <div className="text-xs text-light-muted-text dark:text-dark-muted-text capitalize mt-1 px-2 py-1 bg-light-highlight/10 dark:bg-dark-highlight/10 rounded-full inline-block">
                                                                {user.role?.toLowerCase() ||
                                                                    "User"}
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                                <div className="py-2">
                                                    <motion.button
                                                        onClick={
                                                            handleProfileClick
                                                        }
                                                        className="w-full flex items-center space-x-4 px-5 py-3 text-left text-light-text dark:text-dark-text hover:bg-light-muted-background/50 dark:hover:bg-dark-muted-background/50 transition-all duration-200 group"
                                                        whileHover={{ x: 4 }}
                                                        transition={{
                                                            duration: 0.2,
                                                        }}
                                                    >
                                                        <User className="w-5 h-5 text-light-highlight dark:text-dark-highlight group-hover:scale-110 transition-transform" />
                                                        <span className="font-medium">
                                                            Profile Settings
                                                        </span>
                                                    </motion.button>
                                                    <div className="border-t border-light-muted-text/10 dark:border-dark-muted-text/10 my-2 mx-4"></div>
                                                    <motion.button
                                                        onClick={handleLogout}
                                                        className="w-full flex items-center space-x-4 px-5 py-3 text-left text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 transition-all duration-200 group"
                                                        whileHover={{ x: 4 }}
                                                        transition={{
                                                            duration: 0.2,
                                                        }}
                                                    >
                                                        <LogOut className="w-5 h-5 group-hover:scale-110 transition-transform" />
                                                        <span className="font-medium">
                                                            Sign Out
                                                        </span>
                                                    </motion.button>
                                                </div>
                                            </motion.div>
                                        )}
                                    </AnimatePresence>
                                </div>
                            ) : (
                                <div className="flex items-center space-x-3">
                                    <motion.div
                                        whileHover={{ scale: 1.02 }}
                                        whileTap={{ scale: 0.98 }}
                                    >
                                        <NavLink
                                            to="/auth?view=login"
                                            className="bg-light-muted-background/50 dark:bg-dark-muted-background/50 border border-light-muted-text/20 hover:border-light-highlight dark:hover:border-dark-highlight dark:border-dark-muted-text/20 backdrop-blur-sm font-medium py-2.5 px-6 rounded-full transition-all duration-300 hover:shadow-md"
                                        >
                                            Login
                                        </NavLink>
                                    </motion.div>
                                    <motion.div
                                        whileHover={{ scale: 1.02 }}
                                        whileTap={{ scale: 0.98 }}
                                    >
                                        <NavLink
                                            to="/auth?view=signup"
                                            className="bg-gradient-to-r from-light-highlight to-light-highlight/90 dark:from-dark-highlight dark:to-dark-highlight/90 text-white border border-transparent font-medium py-2.5 px-6 rounded-full transition-all duration-300 hover:shadow-lg shadow-md"
                                        >
                                            Sign Up
                                        </NavLink>
                                    </motion.div>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Mobile Menu Button */}
                    <div className="flex lg:hidden items-center space-x-3">
                        <ThemeToggle />
                        <div className="mobile-menu-container">
                            <motion.button
                                onClick={() =>
                                    setShowMobileMenu(!showMobileMenu)
                                }
                                className="p-2.5 bg-light-muted-background/50 dark:bg-dark-muted-background/50 border border-light-muted-text/20 hover:border-light-highlight dark:hover:border-dark-highlight dark:border-dark-muted-text/20 backdrop-blur-sm rounded-xl transition-all duration-300"
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                            >
                                <motion.div
                                    animate={{
                                        rotate: showMobileMenu ? 90 : 0,
                                    }}
                                    transition={{ duration: 0.2 }}
                                >
                                    {showMobileMenu ? (
                                        <X className="w-5 h-5" />
                                    ) : (
                                        <Menu className="w-5 h-5" />
                                    )}
                                </motion.div>
                            </motion.button>
                        </div>
                    </div>
                </div>
            </motion.header>

            {/* Mobile Menu */}
            <AnimatePresence>
                {showMobileMenu && (
                    <>
                        <motion.div
                            className="lg:hidden fixed inset-0 z-40 bg-black/60 backdrop-blur-sm"
                            variants={overlayVariants}
                            initial="hidden"
                            animate="visible"
                            exit="hidden"
                            onClick={() => setShowMobileMenu(false)}
                        />
                        <motion.div
                            className="mobile-menu-container lg:hidden fixed top-0 right-0 h-full w-80 max-w-[85vw] bg-light-background/95 dark:bg-dark-background/95 backdrop-blur-xl border-l border-light-muted-text/20 dark:border-dark-muted-text/20 shadow-2xl z-50"
                            variants={mobileMenuVariants}
                            initial="hidden"
                            animate="visible"
                            exit="hidden"
                        >
                            <div className="p-5 border-b border-light-muted-text/10 dark:border-dark-muted-text/10 bg-gradient-to-r from-light-muted-background/30 to-transparent dark:from-dark-muted-background/30">
                                <div className="flex items-center justify-between">
                                    <h2 className="text-lg font-bold">
                                        {projectName}
                                    </h2>
                                    <motion.button
                                        onClick={() => setShowMobileMenu(false)}
                                        whileHover={{ scale: 1.1 }}
                                        whileTap={{ scale: 0.9 }}
                                        className="p-1.5 rounded-lg hover:bg-light-muted-background/50 dark:hover:bg-dark-muted-background/50 transition-colors"
                                    >
                                        <X className="w-5 h-5" />
                                    </motion.button>
                                </div>
                            </div>

                            <div className="flex flex-col h-full p-5">
                                <nav className="flex-grow space-y-2">
                                    {(isAuthenticated
                                        ? authenticatedNavItems
                                        : navItems
                                    ).map((item, index) => (
                                        <motion.div
                                            key={item.path}
                                            variants={menuItemVariants}
                                            initial="hidden"
                                            animate="visible"
                                            custom={index}
                                        >
                                            <NavLink
                                                to={item.path}
                                                onClick={handleNavClick}
                                                className={({ isActive }) =>
                                                    `block px-4 py-3 rounded-xl font-medium transition-all duration-200 ${
                                                        isActive
                                                            ? "bg-gradient-to-r from-light-highlight to-light-highlight/80 dark:from-dark-highlight dark:to-dark-highlight/80 text-white shadow-md"
                                                            : "hover:bg-light-muted-background/50 dark:hover:bg-dark-muted-background/50 text-light-text dark:text-dark-text"
                                                    }`
                                                }
                                            >
                                                {item.label}
                                            </NavLink>
                                        </motion.div>
                                    ))}
                                </nav>

                                <motion.div
                                    className="mt-auto pt-5 border-t border-light-muted-text/10 dark:border-dark-muted-text/10"
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.5, duration: 0.3 }}
                                >
                                    {isAuthenticated ? (
                                        <div className="space-y-3">
                                            <motion.button
                                                onClick={handleProfileClick}
                                                className="w-full flex items-center space-x-4 px-4 py-3 text-left rounded-xl hover:bg-light-muted-background/50 dark:hover:bg-dark-muted-background/50 transition-all duration-200 group"
                                                whileHover={{ x: 4 }}
                                                transition={{ duration: 0.2 }}
                                            >
                                                <User className="w-5 h-5 text-light-highlight dark:text-dark-highlight group-hover:scale-110 transition-transform" />
                                                <span className="font-medium">
                                                    Profile Settings
                                                </span>
                                            </motion.button>
                                            <motion.button
                                                onClick={handleLogout}
                                                className="w-full flex items-center space-x-4 px-4 py-3 text-red-600 dark:text-red-400 rounded-xl hover:bg-red-50 dark:hover:bg-red-900/20 transition-all duration-200 group"
                                                whileHover={{ x: 4 }}
                                                transition={{ duration: 0.2 }}
                                            >
                                                <LogOut className="w-5 h-5 group-hover:scale-110 transition-transform" />
                                                <span className="font-medium">
                                                    Sign Out
                                                </span>
                                            </motion.button>
                                        </div>
                                    ) : (
                                        <div className="space-y-3">
                                            <motion.div
                                                whileHover={{ scale: 1.02 }}
                                                whileTap={{ scale: 0.98 }}
                                            >
                                                <NavLink
                                                    to="/auth?view=login"
                                                    onClick={handleNavClick}
                                                    className="block w-full text-center bg-light-muted-background/50 dark:bg-dark-muted-background/50 backdrop-blur-sm font-medium py-3 rounded-xl border border-light-muted-text/20 dark:border-dark-muted-text/20 hover:border-light-highlight dark:hover:border-dark-highlight transition-all duration-300"
                                                >
                                                    Login
                                                </NavLink>
                                            </motion.div>
                                            <motion.div
                                                whileHover={{ scale: 1.02 }}
                                                whileTap={{ scale: 0.98 }}
                                            >
                                                <NavLink
                                                    to="/auth?view=signup"
                                                    onClick={handleNavClick}
                                                    className="block w-full text-center bg-gradient-to-r from-light-highlight to-light-highlight/90 dark:from-dark-highlight dark:to-dark-highlight/90 text-white font-medium py-3 rounded-xl shadow-md hover:shadow-lg transition-all duration-300"
                                                >
                                                    Sign Up
                                                </NavLink>
                                            </motion.div>
                                        </div>
                                    )}
                                </motion.div>
                            </div>
                        </motion.div>
                    </>
                )}
            </AnimatePresence>
        </>
    );
}

export default Header;
