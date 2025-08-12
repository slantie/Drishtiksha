// src/components/layout/Layout.jsx

import React from "react";
import Header from "./Header";
import Footer from "./Footer";

function Layout({ children }) {
    return (
        <div className="font-sans flex flex-col min-h-screen bg-light-muted-background dark:bg-dark-muted-background text-light-text dark:text-dark-text">
            <Header />
            <main className="flex-1 w-full mx-auto p-6">{children}</main>
            <Footer />
        </div>
    );
}

export default Layout;
