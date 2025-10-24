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
    //   displayPath: "/docs/main",
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
      const filename = pathParts[pathParts.length - 1].replace(".md", "").toLowerCase();
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
export const updateImagePaths = (content, filePath) => {
  // Get the depth of the file to calculate relative path
  const depth = (filePath.match(/\//g) || []).length;
  const prefix = depth > 0 ? "../".repeat(depth) : "";

  // Match markdown image syntax: ![alt](path)
  // Replace relative image paths with assets/docs paths
  const imageRegex = /!\[([^\]]*)\]\((?!https?:\/\/)(?!\/\/)(\.\/)?(.*?)\)/g;
  return content.replace(imageRegex, (match, alt, dot, path) => {
    // If path starts with ./, remove it
    let cleanPath = path.startsWith("./") ? path.slice(2) : path;

      // If it's a relative path, convert it to use assets/docs
      if (!cleanPath.startsWith("/")) {
        // Construct the correct path based on file location
        const category = filePath.split("/")[0];
        if (filePath.includes("/")) {
          // File is in a subdirectory
          return `![${alt}](/assets/docs/${category}/${cleanPath})`;
        } else {
          // Root level file
          return `![${alt}](/assets/docs/${cleanPath})`;
        }
      }

      return match;
    }
  );
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
