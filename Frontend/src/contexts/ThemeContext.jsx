// src/contexts/ThemeContext.jsx

import React, { useState, useEffect, useCallback } from "react";
import { themes } from "../constants/themes";
import { ThemeContext } from "./context";

// themeMode: 'light' | 'dark' | 'system'
export const ThemeProvider = ({ children }) => {
  const [themeMode, setThemeMode] = useState("system");
  const [currentTheme, setCurrentTheme] = useState("purple");

  const applyTheme = useCallback((themeName) => {
    const theme = themes[themeName] || themes.purple;
    document.documentElement.style.setProperty(
      "--color-primary",
      theme.colors.primary
    );
  }, []);

  // Helper to get effective dark mode depending on themeMode
  const getEffectiveDark = useCallback(() => {
    const prefersDark = window.matchMedia(
      "(prefers-color-scheme: dark)"
    ).matches;
    if (themeMode === "system") return prefersDark;
    return themeMode === "dark";
  }, [themeMode]);

  // Initialize from storage or fallbacks
  useEffect(() => {
    const storedMode = localStorage.getItem("themeMode");
    const storedDarkMode = localStorage.getItem("darkMode");
    const storedTheme = localStorage.getItem("colorTheme") || "purple";

    let initialMode = "system";
    if (storedMode) initialMode = storedMode;
    else if (storedDarkMode === "true") initialMode = "dark";
    else if (storedDarkMode === "false") initialMode = "light";

    setThemeMode(initialMode);
    setCurrentTheme(storedTheme);
    applyTheme(storedTheme);

    // Apply initial effective dark class
    const prefersDark = window.matchMedia(
      "(prefers-color-scheme: dark)"
    ).matches;
    const isDark =
      initialMode === "system" ? prefersDark : initialMode === "dark";
    if (isDark) document.documentElement.classList.add("dark");
    else document.documentElement.classList.remove("dark");
  }, [applyTheme]);

  // Keep system preference listener when in 'system' mode
  useEffect(() => {
    const mql = window.matchMedia("(prefers-color-scheme: dark)");
    const handleChange = () => {
      if (themeMode === "system") {
        const prefersDark = mql.matches;
        if (prefersDark) document.documentElement.classList.add("dark");
        else document.documentElement.classList.remove("dark");
      }
    };
    mql.addEventListener?.("change", handleChange);
    // call once to ensure correct class
    handleChange();
    return () => mql.removeEventListener?.("change", handleChange);
  }, [themeMode]);

  const setMode = (mode) => {
    setThemeMode(mode);
    localStorage.setItem("themeMode", mode);

    const prefersDark = window.matchMedia(
      "(prefers-color-scheme: dark)"
    ).matches;
    const effectiveDark = mode === "system" ? prefersDark : mode === "dark";
    if (effectiveDark) document.documentElement.classList.add("dark");
    else document.documentElement.classList.remove("dark");
  };

  const toggleDarkMode = () => {
    const prefersDark = window.matchMedia(
      "(prefers-color-scheme: dark)"
    ).matches;
    if (themeMode === "system") {
      // if system, flip to explicit opposite
      setMode(prefersDark ? "light" : "dark");
    } else {
      setMode(themeMode === "dark" ? "light" : "dark");
    }
  };

  const changeTheme = (themeName) => {
    setCurrentTheme(themeName);
    localStorage.setItem("colorTheme", themeName);
    applyTheme(themeName);
  };

  const darkMode = getEffectiveDark();

  return (
    <ThemeContext.Provider
      value={{
        darkMode,
        themeMode,
        setThemeMode: setMode,
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
