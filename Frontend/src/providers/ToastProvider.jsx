// src/providers/ToastProvider.jsx

import React, { useState, useEffect } from "react";
import { Toaster } from "react-hot-toast";

export function ToastProvider({ children }) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <>
      {children}
      {mounted && (
        <Toaster
          position="bottom-right"
          reverseOrder={false}
          gutter={8}
          containerStyle={{
            top: 20,
            right: 20,
            bottom: 20,
            left: 20,
          }}
          // Centralized styling for all toasts
          toastOptions={{
            // Default options for all toast types
            style: {
              borderRadius: "10px",
              background: "var(--toast-bg)",
              color: "var(--toast-text)",
              border: "1px solid var(--toast-border)",
              boxShadow: "var(--toast-shadow)",
            },
            // Specific options for success toasts
            success: {
              duration: 3000,
            },
            // Specific options for error toasts
            error: {
              duration: 5000,
            },
          }}
        />
      )}
    </>
  );
}

export default ToastProvider;
