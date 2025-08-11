import React, { useState, useEffect } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { User, LogOut, Menu, X } from "lucide-react";
import ThemeToggle from "../ThemeToggle";
import { showToast } from "../../utils/toast";

const projectName = import.meta.env.VITE_PROJECT_NAME;
const backendUrl = import.meta.env.VITE_BACKEND_URL;

const navItems = [
  { path: "/", label: "Home" },
  { path: "/Setup", label: "Setup" },
  { path: "/upload", label: "Upload" },
];

function Header() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showMobileMenu, setShowMobileMenu] = useState(false);
  const navigate = useNavigate();

  // Check authentication status on component mount
  useEffect(() => {
    checkAuthStatus();
  }, []);

  const checkAuthStatus = () => {
    try {
      // Check both localStorage and sessionStorage
      const token =
        localStorage.getItem("authToken") ||
        sessionStorage.getItem("authToken");
      const userData =
        localStorage.getItem("user") || sessionStorage.getItem("user");
      const authStatus =
        localStorage.getItem("isAuthenticated") ||
        sessionStorage.getItem("isAuthenticated");

      if (token && userData && authStatus === "true") {
        setIsAuthenticated(true);
        setUser(JSON.parse(userData));
      } else {
        setIsAuthenticated(false);
        setUser(null);
      }
    } catch (error) {
      console.error("Error checking auth status:", error);
      setIsAuthenticated(false);
      setUser(null);
    }
  };

  const handleLogout = async () => {
    try {
      // Get token for logout request
      const token =
        localStorage.getItem("authToken") ||
        sessionStorage.getItem("authToken");

      // Call logout endpoint
      if (token) {
        await fetch(`${backendUrl}/api/auth/logout`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          credentials: "include",
        });
      }

      // Clear all auth data from both storages
      localStorage.removeItem("authToken");
      localStorage.removeItem("user");
      localStorage.removeItem("isAuthenticated");
      sessionStorage.removeItem("authToken");
      sessionStorage.removeItem("user");
      sessionStorage.removeItem("isAuthenticated");

      // Update state
      setIsAuthenticated(false);
      setUser(null);
      setShowUserMenu(false);
      setShowMobileMenu(false);

      // Show success message
      showToast.success("Logged out successfully!");

      // Redirect to home
      navigate("/");
    } catch (error) {
      console.error("Logout error:", error);
      showToast.error("Logout failed. Please try again.");
    }
  };

  const handleProfileClick = () => {
    setShowUserMenu(false);
    setShowMobileMenu(false);
    navigate("/profile");
  };

  const handleNavClick = () => {
    setShowMobileMenu(false);
  };

  // Close menus when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (showUserMenu && !event.target.closest(".user-menu-container")) {
        setShowUserMenu(false);
      }
      if (showMobileMenu && !event.target.closest(".mobile-menu-container")) {
        setShowMobileMenu(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [showUserMenu, showMobileMenu]);

  // Prevent body scroll when mobile menu is open
  useEffect(() => {
    if (showMobileMenu) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "unset";
    }

    return () => {
      document.body.style.overflow = "unset";
    };
  }, [showMobileMenu]);

  return (
    <>
      <header className="p-4 flex justify-between items-center border-b-2 border-light-muted-text/20 dark:border-dark-muted-text/20 relative">
        {/* Left Side - Logo & Project Name */}
        <div
          className="flex items-center space-x-4"
          onClick={() => navigate("/")}
        >
          <div className="flex items-center">
            <img src="/Logo.png" alt="Logo" className="w-8 h-8" />
          </div>
          <h1 className="text-xl md:text-2xl font-semibold tracking-wide hover:text-light-highlight dark:hover:text-dark-highlight truncate">
            {projectName}
          </h1>
        </div>

        {/* Desktop Navigation - Hidden on mobile */}
        <div className="hidden lg:flex items-center space-x-4">
          <nav>
            <ul className="flex space-x-4">
              {navItems.map((item) => (
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

          <div className="flex items-center space-x-2">
            <span className="h-10 border-l-2 border-light-muted-text/20 dark:border-dark-muted-text/20"></span>
          </div>

          <div className="flex items-center space-x-4">
            <ThemeToggle />

            {isAuthenticated && user ? (
              /* Desktop Authenticated User Menu */
              <div className="relative user-menu-container">
                <button
                  onClick={() => setShowUserMenu(!showUserMenu)}
                  className="flex items-center space-x-2 bg-light-muted-background dark:bg-dark-muted-background border-2 border-light-muted-text/20 hover:border-light-highlight dark:hover:border-dark-highlight dark:border-dark-muted-text/20 tracking-wider font-bold py-2 px-4 rounded-lg transition-all duration-300 hover:scale-105"
                >
                  {/* User Avatar or Initial */}
                  {user.avatar ? (
                    <img
                      src={user.avatar}
                      alt="Profile Avatar"
                      className="w-6 h-6 rounded-full object-cover border border-white/20"
                      onError={(e) => {
                        e.target.style.display = "none";
                        e.target.nextSibling.style.display = "flex";
                      }}
                    />
                  ) : null}
                  <div
                    className={`w-6 h-6 bg-gradient-to-br from-light-highlight to-light-highlight/80 dark:from-dark-highlight dark:to-dark-highlight/80 rounded-full flex items-center justify-center text-white text-sm font-bold ${
                      user.avatar ? "hidden" : "flex"
                    }`}
                  >
                    {user.firstName?.charAt(0)?.toUpperCase() || "U"}
                  </div>

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

                {/* Desktop Dropdown Menu */}
                {showUserMenu && (
                  <div className="absolute right-0 mt-2 w-64 bg-light-background dark:bg-dark-background border-2 border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-lg shadow-lg z-50 overflow-hidden">
                    {/* User Info Header */}
                    <div className="px-4 py-3 border-b border-light-muted-text/10 dark:border-dark-muted-text/10 bg-light-muted-background/30 dark:bg-dark-muted-background/30">
                      <div className="flex items-center space-x-3">
                        {/* Dropdown Avatar or Initial */}
                        {user.avatar ? (
                          <img
                            src={user.avatar}
                            alt="Profile Avatar"
                            className="w-10 h-10 rounded-full object-cover border-2 border-white/20 shadow-sm"
                            onError={(e) => {
                              e.target.style.display = "none";
                              e.target.nextSibling.style.display = "flex";
                            }}
                          />
                        ) : null}
                        <div
                          className={`w-10 h-10 bg-gradient-to-br from-light-highlight to-light-highlight/80 dark:from-dark-highlight dark:to-dark-highlight/80 rounded-full flex items-center justify-center text-white font-bold ${
                            user.avatar ? "hidden" : "flex"
                          }`}
                        >
                          {user.firstName?.charAt(0)?.toUpperCase() || "U"}
                        </div>

                        <div>
                          <div className="font-semibold text-light-text dark:text-dark-text">
                            {user.firstName} {user.lastName}
                          </div>
                          <div className="text-xs text-light-muted-text dark:text-dark-muted-text capitalize">
                            {user.role?.toLowerCase() || "User"}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Menu Items */}
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

        {/* Mobile Right Side - Theme Toggle & Menu Button */}
        <div className="flex lg:hidden items-center space-x-3">
          <ThemeToggle />
          <div className="mobile-menu-container">
            <button
              onClick={() => setShowMobileMenu(!showMobileMenu)}
              className="p-2 bg-light-muted-background dark:bg-dark-muted-background border-2 border-light-muted-text/20 hover:border-light-highlight dark:hover:border-dark-highlight dark:border-dark-muted-text/20 rounded-lg transition-all duration-300"
            >
              {showMobileMenu ? (
                <X className="w-5 h-5 text-light-text dark:text-dark-text" />
              ) : (
                <Menu className="w-5 h-5 text-light-text dark:text-dark-text" />
              )}
            </button>
          </div>
        </div>
      </header>

      {/* Mobile Menu Overlay */}
      {showMobileMenu && (
        <div className="lg:hidden fixed inset-0 z-50 bg-black/50 backdrop-blur-sm">
          <div className="mobile-menu-container fixed top-0 right-0 h-full w-80 max-w-[85vw] bg-light-background dark:bg-dark-background border-l-2 border-light-muted-text/20 dark:border-dark-muted-text/20 shadow-2xl transform transition-transform duration-300">
            {/* Mobile Menu Header */}
            <div className="p-4 border-b-2 border-light-muted-text/20 dark:border-dark-muted-text/20 bg-light-muted-background/30 dark:bg-dark-muted-background/30">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <img src="/Logo.png" alt="Logo" className="w-8 h-8" />
                  <h2 className="text-lg font-semibold text-light-text dark:text-dark-text">
                    {projectName}
                  </h2>
                </div>
                <button
                  onClick={() => setShowMobileMenu(false)}
                  className="p-2 hover:bg-light-muted-background dark:hover:bg-dark-muted-background rounded-lg transition-colors duration-200"
                >
                  <X className="w-5 h-5 text-light-text dark:text-dark-text" />
                </button>
              </div>
            </div>

            {/* Mobile Menu Content */}
            <div className="flex flex-col h-full overflow-y-auto pb-20">
              {/* User Section - If Authenticated */}
              {isAuthenticated && user && (
                <div className="p-4 border-b border-light-muted-text/10 dark:border-dark-muted-text/10 bg-light-muted-background/20 dark:bg-dark-muted-background/20">
                  <div className="flex items-center space-x-3 mb-4">
                    {user.avatar ? (
                      <img
                        src={user.avatar}
                        alt="Profile Avatar"
                        className="w-12 h-12 rounded-full object-cover border-2 border-white/20 shadow-sm"
                        onError={(e) => {
                          e.target.style.display = "none";
                          e.target.nextSibling.style.display = "flex";
                        }}
                      />
                    ) : null}
                    <div
                      className={`w-12 h-12 bg-gradient-to-br from-light-highlight to-light-highlight/80 dark:from-dark-highlight dark:to-dark-highlight/80 rounded-full flex items-center justify-center text-white text-lg font-bold ${
                        user.avatar ? "hidden" : "flex"
                      }`}
                    >
                      {user.firstName?.charAt(0)?.toUpperCase() || "U"}
                    </div>

                    <div>
                      <div className="font-semibold text-light-text dark:text-dark-text">
                        {user.firstName} {user.lastName}
                      </div>
                      <div className="text-sm text-light-muted-text dark:text-dark-muted-text capitalize">
                        {user.role?.toLowerCase() || "User"}
                      </div>
                    </div>
                  </div>

                  {/* User Actions */}
                  <div className="space-y-2">
                    <button
                      onClick={handleProfileClick}
                      className="w-full flex items-center space-x-3 px-3 py-2 text-left text-light-text dark:text-dark-text hover:bg-light-muted-background/50 dark:hover:bg-dark-muted-background/50 rounded-lg transition-colors duration-200"
                    >
                      <User className="w-4 h-4" />
                      <span>Profile</span>
                    </button>
                  </div>
                </div>
              )}

              {/* Navigation Links */}
              <div className="p-4">
                <nav>
                  <ul className="space-y-2">
                    {navItems.map((item) => (
                      <li key={item.path}>
                        <NavLink
                          to={item.path}
                          onClick={handleNavClick}
                          className={({ isActive }) =>
                            `block px-4 py-3 rounded-lg text-light-text dark:text-dark-text hover:bg-light-muted-background/50 dark:hover:bg-dark-muted-background/50 transition-colors duration-200 ${
                              isActive
                                ? "bg-light-highlight/10 dark:bg-dark-highlight/10 text-light-highlight dark:text-dark-highlight border-l-4 border-light-highlight dark:border-dark-highlight"
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
              </div>

              {/* Auth Buttons - If Not Authenticated */}
              {!isAuthenticated && (
                <div className="p-4 mt-auto space-y-3">
                  <NavLink
                    to="/auth?view=login"
                    onClick={handleNavClick}
                    className="block w-full text-center bg-light-muted-background dark:bg-dark-muted-background border-2 border-light-muted-text/20 hover:border-light-highlight dark:hover:border-dark-highlight dark:border-dark-muted-text/20 tracking-wider font-bold py-3 px-4 rounded-lg transition-all duration-300"
                  >
                    Login
                  </NavLink>
                  <NavLink
                    to="/auth?view=signup"
                    onClick={handleNavClick}
                    className="block w-full text-center bg-gradient-to-r from-light-highlight to-light-highlight/90 dark:from-dark-highlight dark:to-dark-highlight/90 text-white border-2 border-transparent tracking-wider font-bold py-3 px-4 rounded-lg transition-all duration-300 shadow-lg"
                  >
                    Sign Up
                  </NavLink>
                </div>
              )}

              {/* Logout Button - If Authenticated */}
              {isAuthenticated && (
                <div className="p-4 mt-auto border-t border-light-muted-text/10 dark:border-dark-muted-text/10">
                  <button
                    onClick={handleLogout}
                    className="w-full flex items-center justify-center space-x-3 px-4 py-3 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors duration-200"
                  >
                    <LogOut className="w-4 h-4" />
                    <span>Logout</span>
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default Header;
