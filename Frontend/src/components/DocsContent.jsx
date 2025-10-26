import React, { useState, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import {
  Copy,
  Check,
  ChevronUp,
  Menu,
  ChevronDown,
  FileText,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { updateImagePaths } from "../utils/docsDiscovery";
import "highlight.js/styles/atom-one-dark.css";
import { Button } from "../components/ui/Button";

// Helper function to generate heading ID
const generateHeadingId = (text) => {
  if (typeof text !== "string") return "";
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, "-");
};

const DocsContent = ({ onToggleSidebar }) => {
  const { "*": path } = useParams();
  const navigate = useNavigate();
  const [content, setContent] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [copiedCode, setCopiedCode] = useState(null);
  const [copiedHeading, setCopiedHeading] = useState(null);
  const [copiedFile, setCopiedFile] = useState(false);
  const [showScrollTop, setShowScrollTop] = useState(false);
  const [headings, setHeadings] = useState([]);
  const [activeHeading, setActiveHeading] = useState("");
  const contentRef = useRef(null);
  const scrollContainerRef = useRef(null);
  const tocRef = useRef(null);
  const activeHeadingRef = useRef(null);

  useEffect(() => {
    const loadContent = async () => {
      try {
        setLoading(true);
        setError(null);

        // Redirect to README if no path
        if (!path) {
          navigate("/docs/Overview", { replace: true });
          return;
        }

        // Fetch the markdown file
        const fetchPath = `/docs/${path}.md`;
        console.log("Fetching documentation from:", fetchPath);
        const response = await fetch(fetchPath);
        if (!response.ok) {
          console.error(
            "Failed to fetch documentation:",
            response.status,
            response.statusText
          );
          throw new Error(`File not found: ${fetchPath}`);
        }
        const rawContent = await response.text();
        console.log("Successfully loaded documentation:", path);

        // Parse frontmatter if present
        const frontmatterMatch = rawContent.match(
          /^---\n([\s\S]*?)\n---\n([\s\S]*)$/
        );
        let processedContent = rawContent;

        if (frontmatterMatch) {
          processedContent = frontmatterMatch[2];
        }

        // Update image paths to use assets/docs folder
        processedContent = updateImagePaths(processedContent, path);

        setContent(processedContent);

        // Extract headings for table of contents
        const headingRegex = /^(#{1,3})\s+(.+)$/gm;
        const extractedHeadings = [];
        let match;
        while ((match = headingRegex.exec(processedContent)) !== null) {
          const text = match[2].replace(/`([^`]+)`/g, "$1"); // Remove backticks from heading text
          extractedHeadings.push({
            level: match[1].length,
            text: text,
            id: generateHeadingId(text),
          });
        }
        setHeadings(extractedHeadings);
      } catch (err) {
        console.error("Error loading docs content:", err);
        setError("Documentation not found");
      } finally {
        setLoading(false);
      }
    };

    loadContent();
  }, [path, navigate]);

  // Handle scroll for "back to top" button and active heading
  useEffect(() => {
    const scrollContainer = scrollContainerRef.current;
    if (!scrollContainer) return;

    const handleScroll = () => {
      // Check if we should show scroll-to-top button
      setShowScrollTop(scrollContainer.scrollTop > 400);

      // Update active heading based on scroll position
      if (headings.length > 0) {
        const headingElements = headings
          .map((h) => {
            const el = document.getElementById(h.id);
            return el ? { id: h.id, element: el } : null;
          })
          .filter(Boolean);

        // Find the heading that's currently in view
        // We need to consider the scroll position of the container
        let currentHeading = null;
        for (const { id, element } of headingElements) {
          const rect = element.getBoundingClientRect();
          // Check if the heading is in the upper portion of the viewport
          // (within 200px from the top)
          if (rect.top <= 200) {
            currentHeading = id;
          }
        }

        // If we found a heading and it's different from the current active one, update it
        if (currentHeading && currentHeading !== activeHeading) {
          setActiveHeading(currentHeading);
        } else if (!currentHeading && headingElements.length > 0) {
          // If no heading is above the threshold, set the first one as active
          setActiveHeading(headingElements[0].id);
        }
      }
    };

    // Initial call
    handleScroll();

    scrollContainer.addEventListener("scroll", handleScroll, { passive: true });
    return () => scrollContainer.removeEventListener("scroll", handleScroll);
  }, [headings, activeHeading]);

  // Auto-scroll TOC to keep active heading visible
  useEffect(() => {
    if (activeHeading && activeHeadingRef.current && tocRef.current) {
      const tocContainer = tocRef.current;
      const activeButton = activeHeadingRef.current;

      const tocRect = tocContainer.getBoundingClientRect();
      const buttonRect = activeButton.getBoundingClientRect();

      // Check if the active button is outside the visible area
      if (buttonRect.top < tocRect.top || buttonRect.bottom > tocRect.bottom) {
        // Calculate the scroll position to center the active item
        const scrollTop =
          activeButton.offsetTop -
          tocContainer.offsetTop -
          tocContainer.clientHeight / 2 +
          activeButton.clientHeight / 2;

        tocContainer.scrollTo({
          top: scrollTop,
          behavior: "smooth",
        });
      }
    }
  }, [activeHeading]);

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    setCopiedCode(text);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  const copyHeadingLink = (headingId) => {
    const link = `${window.location.origin}${window.location.pathname}#${headingId}`;
    navigator.clipboard.writeText(link);
    setCopiedHeading(headingId);
    setTimeout(() => setCopiedHeading(null), 2000);
  };

  const copyFileContent = () => {
    navigator.clipboard.writeText(content);
    setCopiedFile(true);
    setTimeout(() => setCopiedFile(false), 2000);
  };

  const scrollToTop = () => {
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollTo({ top: 0, behavior: "smooth" });
    }
  };

  const scrollToHeading = (headingId) => {
    const element = document.getElementById(headingId);
    const scrollContainer = scrollContainerRef.current;

    if (!element || !scrollContainer) {
      console.warn("Cannot scroll: element or container not found", {
        element: !!element,
        scrollContainer: !!scrollContainer,
      });
      return;
    }

    // Get the current scroll position of the container
    const currentScrollTop = scrollContainer.scrollTop;

    // Get the bounding rectangles
    const containerRect = scrollContainer.getBoundingClientRect();
    const elementRect = element.getBoundingClientRect();

    // Calculate the relative position of the element within the container
    // elementRect.top is relative to viewport, containerRect.top is where container starts
    const relativeTop = elementRect.top - containerRect.top;

    // Calculate target scroll position
    // We want the element to be 100px from the top of the container
    const offset = 100;
    const targetScrollTop = currentScrollTop + relativeTop - offset;

    // Perform the scroll
    scrollContainer.scrollTo({
      top: Math.max(0, targetScrollTop),
      behavior: "smooth",
    });

    // Update active heading
    setActiveHeading(headingId);
  };

  if (loading) {
    return (
      <div className="flex-1 p-8 lg:p-12">
        <div className="max-w-4xl mx-auto">
          <div className="animate-pulse space-y-4">
            <div className="h-12 bg-light-hover dark:bg-dark-hover rounded-lg w-3/4"></div>
            <div className="h-4 bg-light-hover dark:bg-dark-hover rounded w-full"></div>
            <div className="h-4 bg-light-hover dark:bg-dark-hover rounded w-5/6"></div>
            <div className="h-4 bg-light-hover dark:bg-dark-hover rounded w-4/5"></div>
            <div className="h-32 bg-light-hover dark:bg-dark-hover rounded-lg mt-6"></div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex-1 p-8 lg:p-12 flex items-center justify-center">
        <div className="text-center max-w-md">
          <div className="w-16 h-16 bg-red-100 dark:bg-red-900/20 rounded-full flex items-center justify-center mx-auto mb-4">
            <Search className="h-8 w-8 text-red-500" />
          </div>
          <h1 className="text-2xl font-bold text-light-text dark:text-dark-text mb-2">
            Documentation Not Found
          </h1>
          <p className="text-light-muted-text dark:text-dark-muted-text mb-6">
            The requested documentation page could not be found. Please check
            the URL or navigate using the sidebar.
          </p>
          <button
            onClick={() => navigate("/docs/Overview")}
            className="px-6 py-2 bg-primary-main text-white rounded-lg hover:bg-primary-dark transition-colors"
          >
            Go to Documentation Home
          </button>
        </div>
      </div>
    );
  }

  return (
    <div
      ref={scrollContainerRef}
      className="flex-1 overflow-y-auto bg-light-background dark:bg-dark-background"
    >
      <div className="mx-auto px-6 py-24 lg:px-12 lg:py-12">
        {/* Mobile menu button */}
        <Button
          onClick={onToggleSidebar}
          className="lg:hidden fixed bottom-6 left-6 z-50 w-12 h-12 bg-primary-main text-white rounded-full shadow-lg flex items-center justify-center hover:bg-primary-dark transition-all"
        >
          <Menu className="h-6 w-6" />
        </Button>

        {/* Global copy button - for README or main documentation */}
        {path === "main" && (
          <div className="mb-6 flex justify-end">
            <Button
              variant="outline"
              onClick={copyFileContent}
              className="flex items-center gap-2 px-3 py-2 rounded-lg bg-light-secondary/20 dark:bg-dark-secondary/20 hover:bg-light-secondary/40 dark:hover:bg-dark-secondary/40 text-light-muted-text dark:text-dark-muted-text hover:text-light-text dark:hover:text-dark-text transition-all duration-200 text-sm font-medium"
              title={copiedFile ? "Copied!" : "Copy entire documentation"}
              type="button"
            >
              {copiedFile ? (
                <>
                  <Check className="h-4 w-4 text-green-500" />
                  <span>Copied</span>
                </>
              ) : (
                <>
                  <FileText className="h-4 w-4" />
                  <span>Copy Documentation</span>
                </>
              )}
            </Button>
          </div>
        )}

        <div className="flex gap-12">
          {/* Main content */}
          <article
            ref={contentRef}
            className="flex-1 min-w-0 prose prose-lg dark:prose-invert max-w-none"
          >
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeHighlight]}
              components={{
                h1: ({ children, ...props }) => {
                  // Extract text for ID generation - handle nested elements and code blocks
                  const extractText = (node) => {
                    if (typeof node === "string") return node;
                    if (Array.isArray(node))
                      return node.map(extractText).join("");
                    if (node?.props?.children)
                      return extractText(node.props.children);
                    return "";
                  };
                  const textContent = extractText(children);
                  const id = generateHeadingId(textContent);

                  return (
                    <h1
                      id={id}
                      className="text-4xl font-bold text-light-text dark:text-dark-text mb-6 mt-0 pb-4 border-b border-light-secondary dark:border-dark-secondary scroll-mt-20 group flex items-center justify-between"
                      {...props}
                    >
                      <span>{children}</span>
                      <Button
                        variant="outline"
                        onClick={copyFileContent}
                        className="flex items-center gap-2"
                        title={
                          copiedFile ? "Copied!" : "Copy entire documentation"
                        }
                        type="button"
                      >
                        {copiedFile ? (
                          <>
                            <Check className="h-4 w-4 text-green-500" />
                            <span>Copied</span>
                          </>
                        ) : (
                          <>
                            <FileText className="h-4 w-4" />
                            <span>Copy Page</span>
                          </>
                        )}
                      </Button>
                    </h1>
                  );
                },
                h2: ({ children, ...props }) => {
                  // Extract text for ID generation - handle nested elements and code blocks
                  const extractText = (node) => {
                    if (typeof node === "string") return node;
                    if (Array.isArray(node))
                      return node.map(extractText).join("");
                    if (node?.props?.children)
                      return extractText(node.props.children);
                    return "";
                  };
                  const textContent = extractText(children);
                  const id = generateHeadingId(textContent);

                  return (
                    <h2
                      id={id}
                      className="text-3xl font-semibold text-light-text dark:text-dark-text mb-4 mt-10 scroll-mt-20 group flex items-center gap-2"
                      {...props}
                    >
                      <span>{children}</span>
                      <Button
                        variant="ghost"
                        onClick={() => copyHeadingLink(id)}
                        className="p-3 hover:bg-light-secondary/60 dark:hover:bg-dark-secondary/60"
                        title={
                          copiedHeading === id
                            ? "Link copied!"
                            : "Copy heading link"
                        }
                        type="button"
                      >
                        {copiedHeading === id ? (
                          <Check className="h-4 w-4 text-green-500" />
                        ) : (
                          <Copy className="h-4 w-4 text-primary-main" />
                        )}
                      </Button>
                    </h2>
                  );
                },
                h3: ({ children, ...props }) => {
                  // Extract text for ID generation - handle nested elements and code blocks
                  const extractText = (node) => {
                    if (typeof node === "string") return node;
                    if (Array.isArray(node))
                      return node.map(extractText).join("");
                    if (node?.props?.children)
                      return extractText(node.props.children);
                    return "";
                  };
                  const textContent = extractText(children);
                  const id = generateHeadingId(textContent);

                  return (
                    <h3
                      id={id}
                      className="text-2xl font-semibold text-light-text dark:text-dark-text mb-3 mt-8 scroll-mt-20 group flex items-center gap-2"
                      {...props}
                    >
                      <span>{children}</span>
                      <Button
                        variant="ghost"
                        onClick={() => copyHeadingLink(id)}
                        className="p-3 hover:bg-light-secondary/60 dark:hover:bg-dark-secondary/60"
                        title={
                          copiedHeading === id
                            ? "Link copied!"
                            : "Copy heading link"
                        }
                        type="button"
                      >
                        {copiedHeading === id ? (
                          <Check className="h-4 w-4 text-green-500" />
                        ) : (
                          <Copy className="h-4 w-4 text-primary-main" />
                        )}
                      </Button>
                    </h3>
                  );
                },
                p: ({ children }) => (
                  <p className="text-light-text dark:text-dark-text mb-5 leading-relaxed text-base">
                    {children}
                  </p>
                ),
                ul: ({ children }) => (
                  <ul className="list-disc list-outside ml-6 text-light-text dark:text-dark-text mb-5 space-y-2">
                    {children}
                  </ul>
                ),
                ol: ({ children }) => (
                  <ol className="list-decimal list-outside ml-6 text-light-text dark:text-dark-text mb-5 space-y-2">
                    {children}
                  </ol>
                ),
                li: ({ children }) => (
                  <li className="text-light-text dark:text-dark-text leading-relaxed">
                    {children}
                  </li>
                ),
                code: ({ inline, children, className }) => {
                  // Handle inline code
                  // In ReactMarkdown: inline code has no className OR inline=true
                  // Code blocks (inside <pre>) have className="language-xxx"
                  if (!className || inline) {
                    return (
                      <code className="bg-light-secondary/50 dark:bg-dark-secondary/50 px-1 py-0.5 rounded text-[0.9em] font-mono text-primary-main inline">
                        {children}
                      </code>
                    );
                  }

                  // Handle code blocks (fenced code with language specifier)
                  const codeString = String(children).replace(/\n$/, "");
                  const language = className
                    ? className.replace("language-", "")
                    : "text";

                  return (
                    <div className="my-6 rounded-lg overflow-hidden border border-light-secondary dark:border-dark-secondary bg-light-background dark:bg-dark-background">
                      {/* Header bar with language badge and copy button */}
                      <div className="flex justify-between items-center px-4 py-3 border-b border-light-secondary dark:border-dark-secondary bg-light-secondary/20 dark:bg-dark-secondary/20">
                        {/* Language badge */}
                        {language && language !== "text" && (
                          <div className="text-xs font-mono font-semibold text-light-muted-text dark:text-dark-muted-text tracking-wide">
                            {language.toLowerCase().split(" ")[1]}
                          </div>
                        )}
                        <div className="flex-1"></div>

                        {/* Copy button */}
                        <Button
                          variant="ghost"
                          onClick={() => copyToClipboard(codeString)}
                          className="flex items-center gap-1.5 p-3 text-xs"
                          title={
                            copiedCode === codeString ? "Copied!" : "Copy code"
                          }
                          type="button"
                        >
                          {copiedCode === codeString ? (
                            <>
                              <Check className="h-3.5 w-3.5 text-green-500" />
                              <span>Code Copied!</span>
                            </>
                          ) : (
                            <>
                              <Copy className="h-3.5 w-3.5" />
                              <span>Copy Code</span>
                            </>
                          )}
                        </Button>
                      </div>

                      {/* Code content with theme-aware colors */}
                      <pre className="m-0 overflow-x-auto text-sm leading-relaxed font-mono !bg-light-background dark:!bg-dark-background text-light-text dark:text-dark-text scrollbar-thin scrollbar-thumb-light-secondary dark:scrollbar-thumb-dark-secondary scrollbar-track-transparent [&_span]:inherit">
                        <code
                          className={className || ""}
                          style={{ background: "transparent" }}
                        >
                          {children}
                        </code>
                      </pre>
                    </div>
                  );
                },
                pre: ({ children }) => {
                  // Pre tags wrap code blocks - we handle the styling in the code component
                  // Return as-is since code component handles all rendering
                  return <>{children}</>;
                },
                hr: () => (
                  <div className="my-8 border-t border-light-secondary dark:border-dark-secondary"></div>
                ),
                blockquote: ({ children }) => (
                  <blockquote className="border-l-4 border-primary-main bg-primary-main/5 dark:bg-primary-main/10 pl-5 py-3 my-6 rounded-r-lg">
                    <div className="text-light-text dark:text-dark-text">
                      {children}
                    </div>
                  </blockquote>
                ),
                table: ({ children }) => (
                  <div className="overflow-x-auto my-6 rounded-lg border border-light-secondary dark:border-dark-secondary">
                    <table className="min-w-full divide-y divide-light-secondary dark:divide-dark-secondary">
                      {children}
                    </table>
                  </div>
                ),
                thead: ({ children }) => (
                  <thead className="bg-light-hover dark:bg-dark-hover">
                    {children}
                  </thead>
                ),
                th: ({ children }) => (
                  <th className="px-6 py-3 text-left text-xs font-semibold text-light-text dark:text-dark-text uppercase tracking-wider">
                    {children}
                  </th>
                ),
                td: ({ children }) => (
                  <td className="px-6 py-4 text-sm text-light-text dark:text-dark-text border-t border-light-secondary dark:border-dark-secondary">
                    {children}
                  </td>
                ),
                a: ({ href, children }) => (
                  <a
                    href={href}
                    className="text-primary-main hover:text-primary-dark dark:hover:text-primary-main underline decoration-primary-main/30 hover:decoration-primary-main transition-colors"
                    target={href?.startsWith("http") ? "_blank" : undefined}
                    rel={
                      href?.startsWith("http")
                        ? "noopener noreferrer"
                        : undefined
                    }
                  >
                    {children}
                  </a>
                ),
                img: ({ src, alt }) => (
                  <img
                    src={src}
                    alt={alt}
                    className="bg-white rounded-xl shadow-sm my-6 border p-4 border-light-secondary dark:border-dark-secondary"
                  />
                ),
              }}
            >
              {content}
            </ReactMarkdown>
          </article>

          {/* Table of Contents - Desktop only */}
          {headings.length > 0 && (
            <aside className="hidden xl:block w-80 flex-shrink-0">
              <div
                className="sticky top-8 max-h-[calc(100vh-4rem)] overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-light-secondary dark:scrollbar-thumb-dark-secondary scrollbar-track-transparent"
                ref={tocRef}
              >
                <div className="mb-4 pb-3 border-b border-light-secondary dark:border-dark-secondary">
                  <h4 className="text-xs font-bold text-light-text dark:text-dark-text uppercase tracking-wider">
                    On this page
                  </h4>
                </div>
                <nav className="space-y-1">
                  {headings.map((heading) => {
                    const isActive = activeHeading === heading.id;
                    const paddingClass =
                      heading.level === 1
                        ? "pl-3"
                        : heading.level === 2
                        ? "pl-3"
                        : "pl-6";

                    return (
                      <button
                        key={heading.id}
                        ref={isActive ? activeHeadingRef : null}
                        type="button"
                        onClick={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          scrollToHeading(heading.id);
                        }}
                        className={`w-full text-left text-xs transition-all duration-200 py-2 border-l-2 hover:pl-4 ${paddingClass} ${
                          isActive
                            ? "border-primary-main text-primary-main font-semibold bg-primary-main/5 dark:bg-primary-main/10 rounded-r"
                            : "border-transparent text-light-muted-text dark:text-dark-muted-text hover:text-light-text dark:hover:text-dark-text hover:border-light-secondary dark:hover:border-dark-secondary"
                        }`}
                        title={heading.text}
                      >
                        <span className="line-clamp-2">{heading.text}</span>
                      </button>
                    );
                  })}
                </nav>
              </div>
            </aside>
          )}
        </div>
      </div>

      {/* Scroll to top button */}
      <AnimatePresence>
        {showScrollTop && (
          <motion.button
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            onClick={scrollToTop}
            className="fixed bottom-6 right-6 w-12 h-12 bg-primary-main text-dark-muted-text dark:text-dark-muted-text rounded-full shadow-lg flex items-center justify-center hover:bg-primary-dark transition-all z-50"
            title="Scroll to top"
          >
            <ChevronUp className="h-6 w-6" />
          </motion.button>
        )}
      </AnimatePresence>
    </div>
  );
};

export default DocsContent;
