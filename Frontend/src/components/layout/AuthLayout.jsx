// src/components/layout/AuthLayout.jsx

import React from "react";

function AuthLayout({ children }) {
    return (
        <div className="flex flex-col min-h-screen bg-light-background dark:bg-dark-background font-sans text-light-text dark:text-dark-text">
            <main className="flex-1 flex items-center justify-center p-4">
                {children}
            </main>
        </div>
    );
}

export default AuthLayout;
