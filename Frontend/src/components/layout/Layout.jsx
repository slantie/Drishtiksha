import React from "react";
import Header from "./Header";
import Footer from "./Footer";

function Layout({ children }) {
  return (
    <div className="min-h-screen bg-light-muted-background dark:bg-dark-background font-sans text-light-text dark:text-dark-text">
      <Header />
      <main className="flex-1 mx-auto p-6 flex items-center justify-center">
        {children}
      </main>
      <Footer />
    </div>
  );
}

export default Layout;
