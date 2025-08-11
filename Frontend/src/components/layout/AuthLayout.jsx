import React from "react";

function AuthLayout({ children }) {
  return (
    <div className="min-h-screen bg-light-background dark:bg-dark-background text-light-text dark:text-dark-text">
      <main className="min-h-screen flex items-center justify-center">
        {children}
      </main>
    </div>
  );
}

export default AuthLayout;