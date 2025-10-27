import React, { useState, useEffect, useCallback } from "react";
import { Link, useLocation } from "react-router-dom";
import {
  ChevronRight,
  FileText,
  BookOpen,
  Menu,
  X,
  FolderOpen,
  Folder,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { discoverDocFiles } from "../utils/docsDiscovery";
import { Button } from "./ui/Button";

const DocsSidebar = () => {
  const location = useLocation();
  const [collapsed, setCollapsed] = useState(false);
  const [expandedSections, setExpandedSections] = useState({
    backend: false,
    frontend: false,
    server: false,
    models: false,
  });
  const [focusedIndex, setFocusedIndex] = useState(-1);
  const [docsStructure, setDocsStructure] = useState({
    backend: {},
    frontend: {},
    server: {},
  });

  // Load and build docs structure dynamically on mount
  useEffect(() => {
    const loadStructure = async () => {
      try {
        const files = await discoverDocFiles();

        // Build a proper structure with labels from manifest
        const structure = {};

        files.forEach((file) => {
          const pathParts = file.path.split("/");

          if (pathParts.length === 1) {
            // Root level file
            structure[file.path] = {
              title: file.title,
              isFile: true,
            };
          } else {
            // Nested file - build the structure recursively
            let current = structure;

            // Navigate through all path parts except the last one (which is the file)
            for (let i = 0; i < pathParts.length - 1; i++) {
              const part = pathParts[i];
              if (!current[part]) {
                current[part] = {};
              }
              current = current[part];
            }

            // Add the file at the final level
            const fileName = pathParts[pathParts.length - 1];
            current[fileName] = {
              title: file.title,
              isFile: true,
              fullPath: file.path,
            };
          }
        });

        setDocsStructure(structure);
      } catch (error) {
        console.error("Failed to load docs structure:", error);
        // Use default structure as fallback
      }
    };

    loadStructure();
  }, []);

  // Flatten navigation items for keyboard navigation
  const getNavigableItems = () => {
    const items = [];
    const addItems = (node, path = "") => {
      Object.keys(node).forEach((key) => {
        const nodeData = node[key];
        const isFile = nodeData && nodeData.isFile === true;

        // Use fullPath from manifest if available, otherwise construct it
        const fullPath = nodeData?.fullPath || (path ? `${path}/${key}` : key);

        if (isFile) {
          // File
          items.push({
            type: "file",
            key,
            path: fullPath,
            displayPath: `/docs/${fullPath}`,
          });
        } else if (typeof nodeData === "object") {
          // Folder
          items.push({
            type: "folder",
            key,
            path: fullPath,
          });
          if (expandedSections[key]) {
            addItems(nodeData, fullPath);
          }
        }
      });
    };
    addItems(docsStructure);
    return items;
  };

  const navigableItems = getNavigableItems();

  const toggleSection = useCallback((section) => {
    setExpandedSections((prev) => ({
      ...prev,
      [section]: !prev[section],
    }));
  }, []);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (collapsed) return; // Only enable keyboard nav when expanded

      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          setFocusedIndex((prev) =>
            prev < navigableItems.length - 1 ? prev + 1 : 0
          );
          break;
        case "ArrowUp":
          e.preventDefault();
          setFocusedIndex((prev) =>
            prev > 0 ? prev - 1 : navigableItems.length - 1
          );
          break;
        case "Enter":
        case " ":
          e.preventDefault();
          if (focusedIndex >= 0 && focusedIndex < navigableItems.length) {
            const item = navigableItems[focusedIndex];
            if (item.type === "file") {
              // Navigate to file
              window.location.href = item.displayPath;
            } else if (item.type === "folder") {
              // Toggle folder expansion
              toggleSection(item.key);
            }
          }
          break;
        case "Escape":
          setFocusedIndex(-1);
          break;
        default:
          break;
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [collapsed, focusedIndex, navigableItems, toggleSection]);

  const renderTree = (node, path = "", level = 0) => {
    return Object.keys(node).map((key) => {
      const nodeData = node[key];
      const isFile = nodeData && nodeData.isFile === true;

      // Use fullPath from manifest if available, otherwise construct it
      const fullPath = nodeData?.fullPath || (path ? `${path}/${key}` : key);

      const isActive = location.pathname === `/docs/${fullPath}`;
      const isExpanded = expandedSections[key];

      const itemIndex = navigableItems.findIndex(
        (item) => item.key === key && item.path === fullPath
      );
      const isFocused = itemIndex === focusedIndex;

      if (isFile) {
        // Use the title from manifest
        const displayName =
          nodeData.title ||
          key.charAt(0).toUpperCase() + key.slice(1).replace(/-/g, " ");

        return (
          <motion.li
            key={fullPath}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            className={`${level > 0 && !collapsed ? "ml-7" : ""}`}
          >
            <Link
              to={`/docs/${fullPath}`}
              className={`group flex items-center ${
                collapsed ? "justify-center" : "gap-3"
              } px-4 py-2.5 rounded-full text-sm transition-all duration-200 relative overflow-hidden ${
                isActive
                  ? "bg-gradient-to-r from-primary-main/10 to-primary-main/5 dark:from-primary-main/20 dark:to-primary-main/10 text-primary-main font-medium border-primary-main"
                  : isFocused
                  ? "bg-primary-main/5 dark:bg-primary-main/10 border border-primary-main/20"
                  : "text-light-text dark:text-dark-text hover:bg-light-hover dark:hover:bg-dark-hover"
              }`}
              title={collapsed ? displayName : undefined}
              onFocus={() => setFocusedIndex(itemIndex)}
              onBlur={() => setFocusedIndex(-1)}
            >
              <FileText className="h-4 w-4 flex-shrink-0 transition-transform group-hover:scale-110" />
              {!collapsed && (
                <span className="flex-1 truncate text-light-text dark:text-dark-text">
                  {displayName}
                </span>
              )}
              {isActive && !collapsed && (
                <motion.div
                  layoutId="activeDoc"
                  className="absolute inset-0 bg-primary-main/5 dark:bg-primary-main/10"
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                />
              )}
            </Link>
          </motion.li>
        );
      } else if (typeof nodeData === "object" && !nodeData.isFile) {
        const folderName =
          key.charAt(0).toUpperCase() + key.slice(1).replace(/-/g, " ");

        return (
          <li key={fullPath} className={`${level > 0 && !collapsed ? "" : ""}`}>
            <Button
              variant="ghost"
              onClick={() => toggleSection(key)}
              className={`w-full flex items-center ${
                collapsed ? "justify-center" : "gap-3"
              } px-4 py-2.5 text-sm font-semibold ${
                isFocused ? "border-primary-main/20" : ""
              }`}
              title={collapsed ? folderName : undefined}
              onFocus={() => setFocusedIndex(itemIndex)}
              onBlur={() => setFocusedIndex(-1)}
            >
              {!collapsed && (
                <motion.div
                  animate={{ rotate: isExpanded ? 90 : 0 }}
                  transition={{ duration: 0.2, type: "spring", stiffness: 200 }}
                >
                  <ChevronRight className="h-4 w-4 flex-shrink-0 text-light-muted-text dark:text-dark-muted-text group-hover:text-primary-main" />
                </motion.div>
              )}
              {isExpanded ? (
                <FolderOpen className="h-4 w-4 flex-shrink-0 text-primary-main" />
              ) : (
                <Folder className="h-4 w-4 flex-shrink-0 text-light-muted-text dark:text-dark-muted-text group-hover:text-primary-main" />
              )}
              {!collapsed && (
                <span className="flex-1 text-left truncate">{folderName}</span>
              )}
            </Button>
            <AnimatePresence>
              {isExpanded && !collapsed && (
                <motion.ul
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.2, ease: "easeInOut" }}
                  className="space-y-1 mt-1 overflow-hidden"
                >
                  {renderTree(nodeData, fullPath, level + 1)}
                </motion.ul>
              )}
            </AnimatePresence>
          </li>
        );
      }
      return null;
    });
  };

  return (
    <motion.aside
      initial={false}
      animate={{ width: collapsed ? "4rem" : "18rem" }}
      transition={{ duration: 0.3, ease: "easeInOut" }}
      className="bg-light-background dark:bg-dark-background border-r border-light-secondary dark:border-dark-secondary h-full overflow-hidden flex flex-col"
      onClick={collapsed ? () => setCollapsed(false) : undefined}
    >
      {/* Header with collapse toggle */}
      <div className="sticky top-0 z-10 bg-gradient-to-b from-light-background to-light-background/95 dark:from-dark-background dark:to-dark-background/95 border-b border-light-secondary dark:border-dark-secondary backdrop-blur-sm">
        <div className="p-4 flex items-center justify-between">
          {!collapsed && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex-1"
            >
              <h2 className="text-lg font-bold text-light-text dark:text-dark-text flex items-center gap-2">
                {/* <BookOpen className="h-5 w-5 text-primary-main" /> */}
                Project Documentation
              </h2>
              <p className="text-xs text-light-muted-text dark:text-dark-muted-text mt-0.5">
                Guides, Explanations & References
              </p>
            </motion.div>
          )}
          <Button
            variant="ghost"
            onClick={() => setCollapsed(!collapsed)}
            className="p-2 group"
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            title={collapsed ? "Expand" : "Collapse"}
          >
            <motion.div
              animate={{ rotate: collapsed ? 180 : 0 }}
              transition={{ duration: 0.3 }}
            >
              {collapsed ? (
                <Menu className="h-5 w-5 text-light-text dark:text-dark-text" />
              ) : (
                <X className="h-5 w-5 text-light-muted-text dark:text-dark-muted-text group-hover:text-primary-main" />
              )}
            </motion.div>
          </Button>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto p-4 scrollbar-thin scrollbar-thumb-light-secondary dark:scrollbar-thumb-dark-secondary scrollbar-track-transparent">
        <ul className="space-y-1">{renderTree(docsStructure)}</ul>
      </nav>

      {/* Collapsed state indicator */}
      {collapsed && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="absolute bottom-8 left-1/2 transform -translate-x-1/2 flex flex-col gap-2"
        >
          <div className="w-1 h-6 bg-primary-main/30 rounded-full mx-auto" />
          <div className="w-1 h-4 bg-primary-main/20 rounded-full mx-auto" />
          <div className="w-1 h-3 bg-primary-main/10 rounded-full mx-auto" />
        </motion.div>
      )}
    </motion.aside>
  );
};

export default DocsSidebar;
