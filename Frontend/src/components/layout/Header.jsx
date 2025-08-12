// src/components/layout/Header.jsx

import React, { useState, useEffect } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { User, LogOut, Menu, X } from "lucide-react";
import ThemeToggle from "../ThemeToggle";
import { useAuth } from "../../hooks/useAuth.js";
import { showToast } from "../../utils/toast";

const projectName = import.meta.env.VITE_PROJECT_NAME;

const navItems = [{ path: "/", label: "Home" }];

const authenticatedNavItems = [{ path: "/dashboard", label: "Dashboard" }];

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
        showToast.success("Logged out successfully!");
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

    return (
        <>
            <header className="p-4 flex justify-between items-center border-b-2 border-light-muted-text/20 dark:border-dark-muted-text/20 relative">
                <div
                    className="flex items-center space-x-4 cursor-pointer"
                    onClick={() => navigate("/")}
                >
                    <img src="/Logo.svg" alt="Logo" className="w-10 h-10" />
                    <h1 className="text-xl md:text-2xl font-semibold tracking-wide hover:text-light-highlight dark:hover:text-dark-highlight truncate">
                        {projectName}
                    </h1>
                </div>

                <div className="hidden lg:flex items-center space-x-4">
                    <nav>
                        <ul className="flex space-x-4">
                            {(isAuthenticated
                                ? authenticatedNavItems
                                : navItems
                            ).map((item) => (
                                <li key={item.path}>
                                    <NavLink
                                        to={item.path}
                                        className={({ isActive }) =>
                                            `relative text-md tracking-wider text-light-text hover:text-light-highlight dark:text-dark-text dark:hover:text-dark-highlight transition-colors duration-300 ${
                                                isActive
                                                    ? "text-light-highlight dark:text-dark-highlight"
                                                    : ""
                                            }`
                                        }
                                    >
                                        {item.label}
                                    </NavLink>
                                </li>
                            ))}
                        </ul>
                    </nav>

                    <span className="h-10 border-l-2 border-light-muted-text/20 dark:border-dark-muted-text/20"></span>

                    <div className="flex items-center space-x-4">
                        <ThemeToggle />
                        {isAuthenticated && user ? (
                            <div className="relative user-menu-container">
                                <button
                                    onClick={() =>
                                        setShowUserMenu(!showUserMenu)
                                    }
                                    className="flex items-center space-x-2 bg-light-muted-background dark:bg-dark-muted-background border-2 border-light-muted-text/20 hover:border-light-highlight dark:hover:border-dark-highlight dark:border-dark-muted-text/20 tracking-wider font-bold py-2 px-4 rounded-lg transition-all duration-300 hover:scale-105"
                                >
                                    {user.avatar ? (
                                        <img
                                            src={user.avatar}
                                            alt="Avatar"
                                            className="w-6 h-6 rounded-full object-cover border border-white/20"
                                        />
                                    ) : (
                                        <div className="w-6 h-6 bg-gradient-to-br from-light-highlight to-light-highlight/80 dark:from-dark-highlight dark:to-dark-highlight/80 rounded-full flex items-center justify-center text-white text-sm font-bold">
                                            {user.firstName
                                                ?.charAt(0)
                                                ?.toUpperCase() || "U"}
                                        </div>
                                    )}
                                    <span className="hidden xl:inline">
                                        {user.firstName} {user.lastName}
                                    </span>
                                    <div
                                        className={`transform transition-transform duration-200 ${
                                            showUserMenu ? "rotate-180" : ""
                                        }`}
                                    >
                                        ▼
                                    </div>
                                </button>
                                {showUserMenu && (
                                    <div className="absolute right-0 mt-2 w-64 bg-light-background dark:bg-dark-background border-2 border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-lg shadow-lg z-50 overflow-hidden">
                                        <div className="px-4 py-3 border-b border-light-muted-text/10 dark:border-dark-muted-text/10 bg-light-muted-background/30 dark:bg-dark-muted-background/30">
                                            <div className="flex items-center space-x-3">
                                                {user.avatar ? (
                                                    <img
                                                        src={user.avatar}
                                                        alt="Avatar"
                                                        className="w-10 h-10 rounded-full object-cover border-2 border-white/20 shadow-sm"
                                                    />
                                                ) : (
                                                    <div className="w-10 h-10 bg-gradient-to-br from-light-highlight to-light-highlight/80 dark:from-dark-highlight dark:to-dark-highlight/80 rounded-full flex items-center justify-center text-white font-bold">
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
                                                    <div className="text-xs text-light-muted-text dark:text-dark-muted-text capitalize">
                                                        {user.role?.toLowerCase() ||
                                                            "User"}
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                        <div className="py-2">
                                            <button
                                                onClick={handleProfileClick}
                                                className="w-full flex items-center space-x-3 px-4 py-2 text-left text-light-text dark:text-dark-text hover:bg-light-muted-background/50 dark:hover:bg-dark-muted-background/50 transition-colors duration-200"
                                            >
                                                <User className="w-4 h-4" />
                                                <span>Profile</span>
                                            </button>
                                            <div className="border-t border-light-muted-text/10 dark:border-dark-muted-text/10 my-2"></div>
                                            <button
                                                onClick={handleLogout}
                                                className="w-full flex items-center space-x-3 px-4 py-2 text-left text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors duration-200"
                                            >
                                                <LogOut className="w-4 h-4" />
                                                <span>Logout</span>
                                            </button>
                                        </div>
                                    </div>
                                )}
                            </div>
                        ) : (
                            <>
                                <NavLink
                                    to="/auth?view=login"
                                    className="bg-light-muted-background dark:bg-dark-muted-background border-2 border-light-muted-text/20 hover:border-light-highlight dark:hover:border-dark-highlight dark:border-dark-muted-text/20 tracking-wider font-bold py-2 px-4 rounded-lg transition-all duration-300 hover:scale-105"
                                >
                                    Login
                                </NavLink>
                                <NavLink
                                    to="/auth?view=signup"
                                    className="bg-gradient-to-r from-light-highlight to-light-highlight/90 dark:from-dark-highlight dark:to-dark-highlight/90 text-white border-2 border-transparent tracking-wider font-bold py-2 px-4 rounded-lg transition-all duration-300 hover:scale-105 shadow-lg hover:shadow-xl"
                                >
                                    Sign Up
                                </NavLink>
                            </>
                        )}
                    </div>
                </div>

                <div className="flex lg:hidden items-center space-x-3">
                    <ThemeToggle />
                    <div className="mobile-menu-container">
                        <button
                            onClick={() => setShowMobileMenu(!showMobileMenu)}
                            className="p-2 bg-light-muted-background dark:bg-dark-muted-background border-2 border-light-muted-text/20 hover:border-light-highlight dark:hover:border-dark-highlight dark:border-dark-muted-text/20 rounded-lg transition-all duration-300"
                        >
                            {showMobileMenu ? (
                                <X className="w-5 h-5" />
                            ) : (
                                <Menu className="w-5 h-5" />
                            )}
                        </button>
                    </div>
                </div>
            </header>

            {showMobileMenu && (
                <div className="lg:hidden fixed inset-0 z-50 bg-black/50 backdrop-blur-sm">
                    <div className="mobile-menu-container fixed top-0 right-0 h-full w-80 max-w-[85vw] bg-light-background dark:bg-dark-background border-l-2 border-light-muted-text/20 dark:border-dark-muted-text/20 shadow-2xl">
                        <div className="p-4 border-b-2 border-light-muted-text/20 dark:border-dark-muted-text/20 flex items-center justify-between">
                            <h2 className="text-lg font-semibold">Menu</h2>
                            <button onClick={() => setShowMobileMenu(false)}>
                                <X className="w-5 h-5" />
                            </button>
                        </div>
                        <div className="flex flex-col h-full p-4">
                            <nav className="flex-grow">
                                <ul className="space-y-2">
                                    {(isAuthenticated
                                        ? authenticatedNavItems
                                        : navItems
                                    ).map((item) => (
                                        <li key={item.path}>
                                            <NavLink
                                                to={item.path}
                                                onClick={handleNavClick}
                                                className={({ isActive }) =>
                                                    `block px-4 py-3 rounded-lg transition-colors ${
                                                        isActive
                                                            ? "bg-light-highlight/10 text-light-highlight"
                                                            : "hover:bg-light-muted-background/50"
                                                    }`
                                                }
                                            >
                                                {item.label}
                                            </NavLink>
                                        </li>
                                    ))}
                                </ul>
                            </nav>
                            <div className="mt-auto">
                                {isAuthenticated ? (
                                    <div className="space-y-2">
                                        <button
                                            onClick={handleProfileClick}
                                            className="w-full flex items-center space-x-3 px-4 py-3 text-left rounded-lg hover:bg-light-muted-background/50"
                                        >
                                            <User className="w-4 h-4" />
                                            <span>Profile</span>
                                        </button>
                                        <button
                                            onClick={handleLogout}
                                            className="w-full flex items-center space-x-3 px-4 py-3 text-red-600 rounded-lg hover:bg-red-50"
                                        >
                                            <LogOut className="w-4 h-4" />
                                            <span>Logout</span>
                                        </button>
                                    </div>
                                ) : (
                                    <div className="space-y-3">
                                        <NavLink
                                            to="/auth?view=login"
                                            onClick={handleNavClick}
                                            className="block w-full text-center bg-light-muted-background font-bold py-3 rounded-lg"
                                        >
                                            Login
                                        </NavLink>
                                        <NavLink
                                            to="/auth?view=signup"
                                            onClick={handleNavClick}
                                            className="block w-full text-center bg-light-highlight text-white font-bold py-3 rounded-lg"
                                        >
                                            Sign Up
                                        </NavLink>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
}

export default Header;
