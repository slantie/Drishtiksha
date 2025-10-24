import React, { useState, useEffect, useRef } from "react";
import { Search, X, FileText, Hash } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "./ui/Button";
import { discoverDocFiles, updateImagePaths } from "../utils/docsDiscovery";

// Helper function to generate heading ID (same as in DocsContent)
const generateHeadingId = (text) => {
  if (typeof text !== "string") return "";
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, "-");
};

const DocsSearch = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [allContent, setAllContent] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const searchRef = useRef(null);
  const inputRef = useRef(null);
  const navigate = useNavigate();

  // Fetch and parse all documentation files on mount
  useEffect(() => {
    const fetchAllDocs = async () => {
      setLoading(true);
      try {
        // Dynamically discover available documentation pages
        const docsPages = await discoverDocFiles();
        const allData = [];

        for (const page of docsPages) {
          try {
            const response = await fetch(`/docs/${page.path}.md`);
            if (response.ok) {
              let content = await response.text();

              // Update image paths to use assets/docs folder
              content = updateImagePaths(content, page.path);

              // Extract headings from content
              const headingRegex = /^(#{1,3})\s+(.+)$/gm;
              const headings = [];
              let match;

              while ((match = headingRegex.exec(content)) !== null) {
                const level = match[1].length;
                const text = match[2].replace(/`([^`]+)`/g, "$1"); // Remove backticks
                const id = generateHeadingId(text);

                headings.push({
                  type: "heading",
                  level,
                  text,
                  id,
                  page: page.title,
                  path: page.displayPath,
                });
              }

              // Add page itself as a result
              allData.push({
                type: "page",
                title: page.title,
                path: page.displayPath,
                headings,
              });

              // Add all headings
              allData.push(...headings);
            }
          } catch (error) {
            console.error(`Failed to fetch ${page.path}:`, error);
          }
        }

        setAllContent(allData);
      } catch (error) {
        console.error("Error discovering documentation files:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchAllDocs();
  }, []);

  // Open search with keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "k") {
        e.preventDefault();
        setIsOpen(true);
      }
      if (e.key === "Escape") {
        setIsOpen(false);
        setSelectedIndex(-1);
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Handle keyboard navigation when search is open
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e) => {
      if (results.length === 0) return;

      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          setSelectedIndex((prev) =>
            prev < results.length - 1 ? prev + 1 : prev
          );
          break;
        case "ArrowUp":
          e.preventDefault();
          setSelectedIndex((prev) => (prev > 0 ? prev - 1 : -1));
          break;
        case "Enter":
          e.preventDefault();
          if (selectedIndex >= 0 && selectedIndex < results.length) {
            const result = results[selectedIndex];
            if (result.type === "page") {
              navigate(result.path);
            } else if (result.type === "heading") {
              navigate(`${result.path}#${result.id}`);
              setTimeout(() => {
                const element = document.getElementById(result.id);
                if (element) {
                  element.scrollIntoView({
                    behavior: "smooth",
                    block: "start",
                  });
                }
              }, 100);
            }
            setIsOpen(false);
            setQuery("");
            setSelectedIndex(-1);
          }
          break;
        default:
          break;
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, results, selectedIndex, navigate]);

  // Focus input when opened
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  // Handle search
  useEffect(() => {
    if (query.trim() === "") {
      setResults([]);
      return;
    }

    const searchQuery = query.toLowerCase();
    const filtered = allContent.filter((item) => {
      if (item.type === "page") {
        return item.title.toLowerCase().includes(searchQuery);
      } else if (item.type === "heading") {
        return item.text.toLowerCase().includes(searchQuery);
      }
      return false;
    });

    // Limit to 15 results for better UX
    setResults(filtered.slice(0, 15));
  }, [query, allContent]);

  const handleSelect = (result) => {
    if (result.type === "page") {
      navigate(result.path);
    } else if (result.type === "heading") {
      // Navigate to the page with the heading anchor
      navigate(`${result.path}#${result.id}`);

      // After navigation, scroll to the heading
      setTimeout(() => {
        const element = document.getElementById(result.id);
        if (element) {
          element.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      }, 100);
    }
    setIsOpen(false);
    setQuery("");
    setSelectedIndex(-1);
  };

  const handleClose = () => {
    setIsOpen(false);
    setQuery("");
    setSelectedIndex(-1);
  };

  return (
    <>
      {/* Search Trigger Button */}
      <Button
        onClick={() => setIsOpen(true)}
        className="hidden sm:flex items-center gap-4 px-3 py-1.5 text-sm text-light-muted-text dark:text-dark-muted-text bg-light-hover dark:bg-dark-hover border border-light-secondary dark:border-dark-secondary rounded-full hover:border-primary-main transition-colors group"
      >
        <Search className="h-4 w-4" />
        <span className="font-normal">Search docs...</span>
        <kbd className="ml-auto px-2 py-0.5 text-sm bg-light-background dark:bg-dark-background border border-light-secondary dark:border-dark-secondary rounded-full">
          Ctrl + K
        </kbd>
      </Button>

      {/* Mobile Search Button */}
      <Button
        onClick={() => setIsOpen(true)}
        className="sm:hidden p-2 text-light-text dark:text-dark-text hover:text-primary-main transition-colors"
      >
        <Search className="h-5 w-5" />
      </Button>

      {/* Search Modal */}
      <AnimatePresence>
        {isOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={handleClose}
              className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40"
            />

            {/* Wrapper that centers the dialog and ensures the backdrop covers full viewport */}
            <div className="fixed inset-0 mr-[20px] z-50 flex items-start justify-center pt-[10vh] pointer-events-none">
              <motion.div
                ref={searchRef}
                initial={{ opacity: 0, scale: 0.95, y: -20 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95, y: -20 }}
                onClick={(e) => e.stopPropagation()}
                className="pointer-events-auto w-full max-w-3xl bg-light-background dark:bg-dark-background rounded-xl shadow-2xl border border-light-secondary dark:border-dark-secondary overflow-hidden"
              >
                {/* Search Input */}
                <div className="flex items-center gap-3 px-4 py-3 border-b border-light-secondary dark:border-dark-secondary">
                  <Search className="h-5 w-5 text-light-muted-text dark:text-dark-muted-text flex-shrink-0" />
                  <input
                    ref={inputRef}
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Search documentation..."
                    className="flex-1 bg-transparent text-light-text dark:text-dark-text placeholder-light-muted-text dark:placeholder-dark-muted-text outline-none text-base"
                  />
                  {query && (
                    <button
                      onClick={() => setQuery("")}
                      className="p-1 hover:bg-light-hover dark:hover:bg-dark-hover rounded transition-colors"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  )}
                  <button
                    onClick={handleClose}
                    className="p-1 hover:bg-light-hover dark:hover:bg-dark-hover rounded transition-colors"
                  >
                    <kbd className="px-2 py-1 text-xs bg-light-hover dark:bg-dark-hover border border-light-secondary dark:border-dark-secondary rounded">
                      ESC
                    </kbd>
                  </button>
                </div>

                {/* Results */}
                <div className="max-h-[60vh] overflow-y-auto">
                  {loading ? (
                    <div className="p-8 text-center">
                      <div className="w-16 h-16 bg-primary-main/10 rounded-full flex items-center justify-center mx-auto mb-4 animate-pulse">
                        <Search className="h-8 w-8 text-primary-main" />
                      </div>
                      <p className="text-light-muted-text dark:text-dark-muted-text">
                        Loading documentation...
                      </p>
                    </div>
                  ) : query === "" ? (
                    <div className="p-8 text-center">
                      <div className="w-16 h-16 bg-primary-main/10 rounded-full flex items-center justify-center mx-auto mb-4">
                        <Search className="h-8 w-8 text-primary-main" />
                      </div>
                      <p className="text-light-muted-text dark:text-dark-muted-text mb-2">
                        Search across all documentation
                      </p>
                      <p className="text-xs text-light-muted-text dark:text-dark-muted-text">
                        Pages, headings, and content
                      </p>
                    </div>
                  ) : results.length > 0 ? (
                    <div className="p-2">
                      {results.map((result, index) => (
                        <button
                          key={`${result.type}-${result.path || ""}-${
                            result.id || ""
                          }-${index}`}
                          onClick={() => handleSelect(result)}
                          className={`w-full flex items-start gap-3 p-3 rounded-lg transition-colors text-left group ${
                            index === selectedIndex
                              ? "bg-primary-main/10 dark:bg-primary-main/20 border border-primary-main/20"
                              : "hover:bg-light-hover dark:hover:bg-dark-hover"
                          }`}
                        >
                          {result.type === "page" ? (
                            <>
                              <div className="p-2 bg-primary-main/10 rounded-lg flex-shrink-0">
                                <FileText className="h-5 w-5 text-primary-main" />
                              </div>
                              <div className="flex-1 min-w-0">
                                <div className="font-medium text-light-text dark:text-dark-text group-hover:text-primary-main transition-colors">
                                  {result.title}
                                </div>
                                <div className="text-xs text-light-muted-text dark:text-dark-muted-text">
                                  Documentation Page
                                </div>
                              </div>
                            </>
                          ) : (
                            <>
                              <div className="p-2 bg-primary-main/10 dark:bg-primary-main/20 rounded-lg flex-shrink-0">
                                <Hash className="h-5 w-5 text-primary-main" />
                              </div>
                              <div className="flex-1 min-w-0">
                                <div className="font-medium text-light-text dark:text-dark-text group-hover:text-primary-main transition-colors">
                                  {result.text}
                                </div>
                                <div className="flex items-center gap-2 text-xs text-light-muted-text dark:text-dark-muted-text">
                                  <span className="px-1.5 py-0.5 bg-light-hover dark:bg-dark-hover rounded">
                                    H{result.level}
                                  </span>
                                  <span>in {result.page}</span>
                                </div>
                              </div>
                            </>
                          )}
                        </button>
                      ))}
                    </div>
                  ) : (
                    <div className="p-8 text-center">
                      <div className="w-16 h-16 bg-light-hover dark:bg-dark-hover rounded-full flex items-center justify-center mx-auto mb-4">
                        <Search className="h-8 w-8 text-light-muted-text dark:text-dark-muted-text" />
                      </div>
                      <p className="text-light-muted-text dark:text-dark-muted-text">
                        No results found for "{query}"
                      </p>
                    </div>
                  )}
                </div>

                {/* Footer */}
                <div className="px-4 py-3 border-t border-light-secondary dark:border-dark-secondary bg-light-hover/50 dark:bg-dark-hover/50">
                  <div className="flex items-center justify-between text-xs text-light-muted-text dark:text-dark-muted-text">
                    <div className="flex items-center gap-4">
                      <span className="flex items-center gap-1">
                        <kbd className="px-1.5 py-0.5 bg-light-background dark:bg-dark-background border border-light-secondary dark:border-dark-secondary rounded">
                          ↑
                        </kbd>
                        <kbd className="px-1.5 py-0.5 bg-light-background dark:bg-dark-background border border-light-secondary dark:border-dark-secondary rounded">
                          ↓
                        </kbd>
                        <span>Navigate</span>
                      </span>
                      <span className="flex items-center gap-1">
                        <kbd className="px-1.5 py-0.5 bg-light-background dark:bg-dark-background border border-light-secondary dark:border-dark-secondary rounded">
                          ↵
                        </kbd>
                        <span>Select</span>
                      </span>
                    </div>
                  </div>
                </div>
              </motion.div>
            </div>
          </>
        )}
      </AnimatePresence>
    </>
  );
};

export default DocsSearch;
