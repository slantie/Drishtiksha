/**
 * Documentation Discovery Utility
 * Automatically discovers and indexes all markdown files in the docs folder
 * Resilient to new file additions without code changes
 */

/**
 * Recursively discover all markdown files in the docs folder
 * @returns {Promise<Array>} Array of discovered documentation pages
 */
export const discoverDocFiles = async () => {
  try {
    // We'll use a manifest file or dynamic discovery
    // For now, we'll try to fetch a manifest that lists all docs
    try {
      const manifest = await fetch("/docs/manifest.json");
      if (manifest.ok) {
        return await manifest.json();
      }
    } catch {
      console.warn("Manifest not found, using fallback discovery");
    }

    // Fallback: Use known directories and attempt to discover files
    // This is a static fallback that can be manually maintained
    return getStaticDocsList();
  } catch (error) {
    console.error("Error discovering doc files:", error);
    return getStaticDocsList();
  }
};

/**
 * Get static list of documentation pages
 * This can be maintained as fallback or primary source
 * @returns {Array} Array of documentation pages
 */
export const getStaticDocsList = () => {
  return [
    // {
    //   title: "Getting Started",
    //   path: "main",
    //   displayPath: "/docs/Overview",
    //   category: "root",
    // },
    {
      title: "Backend Documentation",
      path: "backend/main",
      displayPath: "/docs/backend/main",
      category: "backend",
    },
    {
      title: "Frontend Documentation",
      path: "frontend/main",
      displayPath: "/docs/frontend/main",
      category: "frontend",
    },
    {
      title: "Server Documentation",
      path: "server/main",
      displayPath: "/docs/server/main",
      category: "server",
    },
  ];
};

/**
 * Build sidebar structure from discovered files
 * Automatically organizes files by directory
 * @param {Array} files - Array of discovered files
 * @returns {Object} Sidebar structure
 */
export const buildSidebarStructure = (files) => {
  const structure = {
    README: null, // Root level
  };

  // Group files by category
  const categories = {};

  files.forEach((file) => {
    const pathParts = file.path.split("/");
    if (pathParts.length === 1) {
      // Root level file
      const key = pathParts[0].replace(".md", "").toLowerCase();
      if (key !== "main") {
        structure[key] = null;
      }
    } else {
      // Categorized file
      const category = pathParts[0];
      if (!categories[category]) {
        categories[category] = {};
      }
      const filename = pathParts[pathParts.length - 1]
        .replace(".md", "")
        .toLowerCase();
      categories[category][filename] = null;
    }
  });

  // Add categories to structure
  Object.assign(structure, categories);

  return structure;
};

/**
 * Update image paths in markdown content to use assets/docs folder
 * Handles relative paths and converts them appropriately
 * @param {string} content - Markdown content
 * @param {string} filePath - Current file path (e.g., "backend/main")
 * @returns {string} Updated markdown with fixed image paths
 */
// utils/docsDiscovery.js
export const updateImagePaths = (content, filePath = "") => {
  if (!content) return content;

  // -----------------------------------------------------------------
  // Helper – turn a relative image path into the final public URL
  // -----------------------------------------------------------------
  const makeAssetUrl = (relPath) => {
    if (typeof relPath !== "string" || relPath.trim() === "") return undefined;

    // Strip leading ./ or .\
    const clean = relPath.replace(/^(\.\/|\.\\)/, "").trim();
    if (!clean) return undefined;

    const parts = filePath.split("/");
    const isRoot = parts.length <= 1;
    const category = isRoot ? "" : parts[0];

    const base = isRoot ? "/assets/docs" : `/assets/docs/${category}`;
    console.log(`${base}/${clean}`.replace(/\/+/g, "/"));
    return `${base}/${clean}`.replace(/\/+/g, "/"); // no double slashes
  };

  // -----------------------------------------------------------------
  // 1. Markdown images:  ![alt](relative/path.jpg)
  // -----------------------------------------------------------------
  const mdRegex = /!\[([^\]]*)\]\((?!https?:\/\/)(?!data:)(?!\/)(.*?)\)/g;
  content = content.replace(mdRegex, (match, alt, relPath) => {
    const url = makeAssetUrl(relPath);
    return url ? `![${alt}](${url})` : match;
  });

  // -----------------------------------------------------------------
  // 2. HTML <img> tags:  <img ... src="relative/path.jpg" …>
  // -----------------------------------------------------------------
  const htmlRegex =
    /<img\s+([^>]*?)src=["'](?!(https?:\/\/|data:|\/))(.*?)(["'][^>]*?)>/gi;
  content = content.replace(htmlRegex, (match, before, relPath, after) => {
    const url = makeAssetUrl(relPath);
    return url ? `<img ${before}src="${url}"${after}>` : match;
  });

  return content;
};

/**
 * Helper function to get the correct image base path for a document
 * @param {string} filePath - Document path (e.g., "backend/main" or "main")
 * @returns {string} Base path for images
 */
export const getImageBasePath = (filePath) => {
  const pathParts = filePath.split("/");
  if (pathParts.length > 1) {
    // Nested file - use category directory
    return `/assets/docs/${pathParts[0]}`;
  }
  // Root level file
  return "/assets/docs";
};

/**
 * Generate a manifest of all documentation files
 * This can be used to pre-generate a manifest.json file
 * @returns {string} JSON string for manifest
 */
export const generateManifest = (files) => {
  return JSON.stringify(files, null, 2);
};
