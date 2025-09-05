// src/components/layout/AuthLayout.jsx

import React from "react";
import ThemeToggle from "../ThemeToggle";

function AuthLayout({ children }) {
  // REFACTOR: This component is now a generic layout container.
  // Its sole responsibility is to provide the background and center the content passed to it.
  return (
    <div className="flex min-h-screen w-full items-center justify-center bg-light-muted-background dark:bg-dark-background p-4 font-sans text-light-text dark:text-dark-text">
      <div className="fixed top-4 right-4 z-50">
        <ThemeToggle />
      </div>
      {children}
    </div>
  );
}

export default AuthLayout;
