import React from "react";
import Header from "./Header";
import Footer from "./Footer";

function Layout({ children }) {
  return (
    <div className="flex flex-col min-h-screen bg-light-background dark:bg-dark-background text-light-text dark:text-dark-text">
      <Header />
      <main className="flex-1 container mx-auto p-4 flex items-center justify-center">
        {children}
      </main>
      <Footer />
    </div>
  );
}

export default Layout;
