// src/components/ThemeToggle.jsx

import React, { useState, useRef, useEffect } from "react";
import { Sun, Moon, Palette } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "./ui/Button";
import { useTheme } from "../hooks/useTheme";
import { themes } from "../constants/themes";

function ThemeToggle() {
  const { darkMode, currentTheme, toggleDarkMode, changeTheme } = useTheme();
  const [showThemeMenu, setShowThemeMenu] = useState(false);
  const menuRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setShowThemeMenu(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const iconVariants = {
    hidden: { rotate: -90, opacity: 0, scale: 0.5 },
    visible: { rotate: 0, opacity: 1, scale: 1 },
    exit: { rotate: 90, opacity: 0, scale: 0.5 },
  };

  return (
    <div className="flex items-center gap-2">
      {/* Theme Color Selector */}
      <div className="relative" ref={menuRef}>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setShowThemeMenu(!showThemeMenu)}
          aria-label="Change theme color"
        >
          <Palette className="h-5 w-5" />
        </Button>

        <AnimatePresence>
          {showThemeMenu && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: -10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: -10 }}
              transition={{ duration: 0.2 }}
              className="absolute right-0 mt-2 w-48 bg-light-background dark:bg-dark-background border border-light-secondary dark:border-dark-secondary rounded-lg shadow-lg overflow-hidden z-50"
            >
              <div className="p-2">
                <p className="text-xs font-semibold text-light-muted-text dark:text-dark-muted-text px-2 py-1">
                  Color Theme
                </p>
                {Object.values(themes).map((theme) => (
                  <button
                    key={theme.value}
                    onClick={() => {
                      changeTheme(theme.value);
                      setShowThemeMenu(false);
                    }}
                    className={`w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors ${
                      currentTheme === theme.value
                        ? "bg-primary-main/10 dark:bg-primary-main/20 text-primary-main font-medium"
                        : "text-light-text dark:text-dark-text hover:bg-light-hover dark:hover:bg-dark-hover"
                    }`}
                  >
                    <div
                      className="w-4 h-4 rounded-full border-2 border-light-secondary dark:border-dark-secondary"
                      style={{ backgroundColor: theme.colors.primary }}
                    />
                    <span className="flex-1 text-left">{theme.name}</span>
                    {currentTheme === theme.value && (
                      <motion.div
                        layoutId="activeTheme"
                        className="w-1.5 h-1.5 rounded-full bg-primary-main"
                      />
                    )}
                  </button>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Dark/Light Mode Toggle */}
      <Button
        variant="ghost"
        size="icon"
        onClick={toggleDarkMode}
        aria-label={darkMode ? "Switch to light mode" : "Switch to dark mode"}
      >
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={darkMode ? "moon" : "sun"}
            variants={iconVariants}
            initial="hidden"
            animate="visible"
            exit="exit"
            transition={{ duration: 0.2 }}
          >
            {darkMode ? (
              <Sun className="h-5 w-5" />
            ) : (
              <Moon className="h-5 w-5" />
            )}
          </motion.div>
        </AnimatePresence>
      </Button>
    </div>
  );
}

export default ThemeToggle;
