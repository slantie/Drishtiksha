// src/contexts/ThemeContext.jsx

import React, { useState, useEffect } from "react";
import { themes } from "../constants/themes";
import { ThemeContext } from "./context";

export const ThemeProvider = ({ children }) => {
  const [darkMode, setDarkMode] = useState(false);
  const [currentTheme, setCurrentTheme] = useState("purple");

  useEffect(() => {
    // Load theme preferences
    const storedDarkMode = localStorage.getItem("darkMode");
    const storedTheme = localStorage.getItem("colorTheme") || "purple";
    const prefersDark = window.matchMedia(
      "(prefers-color-scheme: dark)"
    ).matches;

    const isDark =
      storedDarkMode === "true" || (!storedDarkMode && prefersDark);
    setDarkMode(isDark);
    setCurrentTheme(storedTheme);

    // Apply dark mode class
    if (isDark) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }

    // Apply theme colors
    applyTheme(storedTheme);
  }, []);

  const applyTheme = (themeName) => {
    const theme = themes[themeName] || themes.purple;
    document.documentElement.style.setProperty(
      "--color-primary",
      theme.colors.primary
    );
  };

  const toggleDarkMode = () => {
    setDarkMode((prev) => {
      const newMode = !prev;
      if (newMode) {
        document.documentElement.classList.add("dark");
        localStorage.setItem("darkMode", "true");
      } else {
        document.documentElement.classList.remove("dark");
        localStorage.setItem("darkMode", "false");
      }
      return newMode;
    });
  };

  const changeTheme = (themeName) => {
    setCurrentTheme(themeName);
    localStorage.setItem("colorTheme", themeName);
    applyTheme(themeName);
  };

  return (
    <ThemeContext.Provider
      value={{
        darkMode,
        currentTheme,
        themes,
        toggleDarkMode,
        changeTheme,
      }}
    >
      {children}
    </ThemeContext.Provider>
  );
};
