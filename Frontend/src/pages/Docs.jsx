import React, { useState } from "react";
import { X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import DocsSidebar from "../components/DocsSidebar";
import DocsContent from "../components/DocsContent";

const Docs = () => {
  const [showSidebar, setShowSidebar] = useState(false);

  const toggleSidebar = () => setShowSidebar(!showSidebar);

  return (
    <div className="flex h-full bg-light-background dark:bg-dark-background">
      {/* Main Content Area */}
      <div className="flex flex-1 overflow-hidden">
        {/* Desktop Sidebar */}
        <div className="hidden lg:block">
          <DocsSidebar />
        </div>

        {/* Mobile Sidebar Overlay */}
        <AnimatePresence>
          {showSidebar && (
            <>
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={toggleSidebar}
                className="lg:hidden fixed inset-0 bg-black/50 backdrop-blur-sm z-40"
              />
              <motion.div
                initial={{ x: "-100%" }}
                animate={{ x: 0 }}
                exit={{ x: "-100%" }}
                transition={{ type: "spring", stiffness: 300, damping: 30 }}
                className="lg:hidden fixed left-0 top-0 bottom-0 z-50 w-80 max-w-[85vw]"
              >
                <div className="h-full bg-light-background dark:bg-dark-background border-r border-light-secondary dark:border-dark-secondary flex flex-col">
                  <div className="flex items-center justify-between p-4 border-b border-light-secondary dark:border-dark-secondary">
                    <h2 className="text-lg font-bold text-light-text dark:text-dark-text">
                      Documentation
                    </h2>
                    <button
                      onClick={toggleSidebar}
                      className="p-2 hover:bg-light-hover dark:hover:bg-dark-hover rounded-lg transition-colors"
                      aria-label="Close sidebar"
                    >
                      <X className="h-5 w-5" />
                    </button>
                  </div>
                  <div className="flex-1 overflow-y-auto">
                    <DocsSidebar />
                  </div>
                </div>
              </motion.div>
            </>
          )}
        </AnimatePresence>

        {/* Content */}
        <DocsContent onToggleSidebar={toggleSidebar} />
      </div>
    </div>
  );
};

export default Docs;
