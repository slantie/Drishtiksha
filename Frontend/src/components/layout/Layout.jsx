// src/components/layout/Layout.jsx

import React from "react";
import { useLocation } from "react-router-dom";
import Header from "./Header";
import Footer from "./Footer";

function Layout({ children }) {
  const location = useLocation();
  const isDocsPage = location.pathname.startsWith("/docs");

  return (
    <div
      className={`flex flex-col ${
        isDocsPage ? "h-screen overflow-hidden" : "min-h-screen"
      } bg-light-muted-background dark:bg-dark-background text-light-text dark:text-dark-text font-sans`}
    >
      <Header />
      <main
        className={`flex-1 w-full mx-auto ${
          isDocsPage ? "overflow-hidden" : "p-4"
        }`}
      >
        {children}
      </main>
      {!isDocsPage && <Footer />}
    </div>
  );
}

export default Layout;
