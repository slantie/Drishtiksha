// src/components/layout/Header.jsx

import React, { useState, useEffect, useRef } from "react";
import { NavLink, useNavigate, useLocation } from "react-router-dom";
import { User, LogOut, Menu, X, ChevronDown } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import ThemeToggle from "../ThemeToggle";
import DocsSearch from "../DocsSearch";
import { useAuth } from "../../hooks/useAuth.js";
import { Button } from "../ui/Button";
import { config } from "../../config/env.js";

const projectName = config.VITE_PROJECT_NAME || "Drishtiksha";

// Define navigation items
const navItems = [{ path: "/docs/main", label: "Documentation" }]; // Public routes

const authenticatedNavItems = [
  { path: "/dashboard", label: "Dashboard" },
  { path: "/monitor", label: "System Monitoring" },
  { path: "/docs/Overview", label: "Documentation" },
  // { path: "/profile", label: "Profile" },
];

function Header() {
  const { isAuthenticated, user, logout } = useAuth();
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showMobileMenu, setShowMobileMenu] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  const userMenuRef = useRef(null); // Ref for user dropdown menu
  const mobileMenuRef = useRef(null); // Ref for mobile menu panel

  // Check if we're on a docs page
  const isDocsPage = location.pathname.startsWith("/docs");

  // Handle scroll effect for sticky header
  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 10);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  // Handle click outside for closing menus
  useEffect(() => {
    const handleClickOutside = (event) => {
      // Close user menu if click is outside
      if (userMenuRef.current && !userMenuRef.current.contains(event.target)) {
        setShowUserMenu(false);
      }
      // Close mobile menu if click is outside its panel
      if (
        mobileMenuRef.current &&
        !mobileMenuRef.current.contains(event.target)
      ) {
        setShowMobileMenu(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []); // Depend on nothing to attach once

  const handleProfileClick = () => {
    setShowUserMenu(false);
    setShowMobileMenu(false);
    navigate("/profile");
  };

  const handleLogoutClick = () => {
    setShowUserMenu(false);
    setShowMobileMenu(false);
    logout(); // Trigger logout from AuthContext
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

  // Avatar display logic
  const renderUserAvatar = () => {
    if (!user) return null; // Should not happen if isAuthenticated is true

    let initials = "";
    if (user.firstName && user.lastName) {
      initials = `${user.firstName.charAt(0)}${user.lastName.charAt(
        0
      )}`.toUpperCase();
    } else if (user.email) {
      initials = user.email.charAt(0).toUpperCase();
    }

    return (
      <div className="w-10 h-10 bg-primary-main/10 rounded-full flex items-center justify-center text-primary-main font-bold text-lg">
        {user.avatar ? (
          <img
            src={user.avatar}
            alt="User Avatar"
            className="w-full h-full rounded-full object-cover"
          />
        ) : (
          initials
        )}
      </div>
    );
  };

  return (
    <header
      className={`sticky top-0 z-50 transition-all duration-300 border-b-2 border-light-noisy-background/40 dark:border-dark-muted-background ${
        scrolled
          ? "border-b border-light-secondary dark:border-dark-secondary bg-light-background dark:bg-dark-background "
          : ""
      }`}
    >
      <div className="mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-20">
          {/* Logo */}
          <div
            className="flex items-center space-x-3 cursor-pointer"
            onClick={() => navigate("/")}
            aria-label={`${projectName} Home`}
          >
            <img src="/Logo.svg" alt="Logo" className="w-10 h-10" />
            <span className="text-xl font-bold tracking-tight">
              {projectName}
            </span>
          </div>

          <div className="flex items-center gap-6">
            {/* Desktop Navigation */}
            <nav className="hidden lg:flex items-center space-x-8">
              {(isAuthenticated ? authenticatedNavItems : navItems).map(
                (item) => (
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
                )
              )}
            </nav>

            {/* Actions */}
            <div className="hidden lg:flex items-center space-x-2">
              {/* Show DocsSearch on docs pages */}
              {isDocsPage && <DocsSearch />}
              <ThemeToggle />
              {isAuthenticated && user ? (
                <div className="relative" ref={userMenuRef}>
                  {" "}
                  {/* Attach ref here */}
                  <button
                    onClick={() => setShowUserMenu(!showUserMenu)}
                    className="flex items-center space-x-2 focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-primary-main rounded-full"
                    aria-haspopup="menu"
                    aria-expanded={showUserMenu ? "true" : "false"}
                    aria-label="User menu"
                  >
                    {renderUserAvatar()}
                    <ChevronDown
                      className={`h-4 w-4 transition-transform duration-200 ${
                        showUserMenu ? "rotate-180" : ""
                      }`}
                    />
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
                            {user.firstName} {user.lastName}
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
                            <User className="h-4 w-4" /> Profile{" "}
                          </button>
                          <button
                            onClick={handleLogoutClick}
                            className="w-full text-left flex items-center gap-2 p-2 rounded-md hover:bg-light-hover dark:hover:bg-dark-hover text-red-500"
                          >
                            {" "}
                            <LogOut className="h-4 w-4" /> Sign Out{" "}
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
                  <Button onClick={() => navigate("/auth?view=signup")}>
                    Sign Up
                  </Button>
                </div>
              )}
            </div>
          </div>

          {/* Mobile Menu Button */}
          <div className="lg:hidden flex items-center gap-2">
            {/* Show DocsSearch on mobile for docs pages */}
            {isDocsPage && <DocsSearch />}
            <ThemeToggle />
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setShowMobileMenu(true)}
              aria-label="Open mobile menu"
            >
              <Menu className="h-6 w-6" />
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
            className="lg:hidden fixed inset-0 z-50"
            onClick={() => setShowMobileMenu(false)} // Close when clicking outside overlay
          >
            <motion.div
              ref={mobileMenuRef}
              initial={{ x: "100%" }}
              animate={{ x: 0 }}
              exit={{ x: "100%" }}
              transition={{
                type: "spring",
                stiffness: 300,
                damping: 30,
              }}
              className="absolute top-0 right-0 h-full w-80 bg-light-background dark:bg-dark-background p-6"
              onClick={(e) => e.stopPropagation()} // Prevent closing when clicking inside panel
            >
              <div className="flex justify-between items-center mb-8">
                <div
                  className="flex items-center space-x-3 cursor-pointer"
                  onClick={() => {
                    setShowMobileMenu(false);
                    navigate("/");
                  }}
                  aria-label={`${projectName} Home`}
                >
                  <img src="/Logo.svg" alt="Logo" className="w-10 h-10" />
                  <span className="text-lg font-bold">{projectName}</span>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setShowMobileMenu(false)}
                  aria-label="Close mobile menu"
                >
                  <X className="h-6 w-6" />
                </Button>
              </div>
              <nav className="flex flex-col space-y-4">
                {(isAuthenticated ? authenticatedNavItems : navItems).map(
                  (item) => (
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
                  )
                )}
                {/* Add login/signup buttons for mobile if not authenticated */}
                {!isAuthenticated && (
                  <>
                    <Button
                      variant="outline"
                      className="w-full justify-start mt-4"
                      onClick={() => {
                        setShowMobileMenu(false);
                        navigate("/auth?view=login");
                      }}
                    >
                      Login
                    </Button>
                    <Button
                      className="w-full justify-start"
                      onClick={() => {
                        setShowMobileMenu(false);
                        navigate("/auth?view=signup");
                      }}
                    >
                      Sign Up
                    </Button>
                  </>
                )}
                {/* Add profile/logout for mobile if authenticated */}
                {isAuthenticated && (
                  <div className="mt-4 pt-4 border-t border-light-secondary dark:border-dark-secondary">
                    {/* <button
                      onClick={handleProfileClick}
                      className="w-full text-left flex items-center gap-2 p-2 rounded-md hover:bg-light-hover dark:hover:bg-dark-hover text-md font-semibold"
                    >
                      <User className="h-5 w-5" /> Profile
                    </button> */}
                    <button
                      onClick={handleLogoutClick}
                      className="w-full text-left flex items-center gap-2 p-2 rounded-md hover:bg-light-hover dark:hover:bg-dark-hover text-red-500 text-md font-semibold"
                    >
                      <LogOut className="h-5 w-5" />
                      Sign Out
                    </button>
                  </div>
                )}
              </nav>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
}

export default Header;
