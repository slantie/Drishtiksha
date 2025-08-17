// src/components/layout/Layout.jsx

import React from "react";
import Header from "./Header";
import Footer from "./Footer";

function Layout({ children }) {
    return (
        <div className="flex flex-col min-h-screen bg-light-muted-background dark:bg-dark-background text-light-text dark:text-dark-text font-sans">
            <Header />
            <main className="flex-1 w-full mx-auto p-6">
                {children}
            </main>
            <Footer />
        </div>
    );
}

export default Layout;
