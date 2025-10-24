/**
 * Documentation Manifest Generator
 * 
 * Automatically scans the docs folder and generates manifest.json
 * Run once or during build to sync documentation files.
 * 
 * Usage: node scripts/generate-docs-manifest.js
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DOCS_SOURCE_DIR = path.join(__dirname, '..', 'docs');
const DOCS_PUBLIC_DIR = path.join(__dirname, '..', 'public', 'docs');
const MANIFEST_PATH = path.join(DOCS_PUBLIC_DIR, 'manifest.json');

/**
 * Extract the first H1 heading from a markdown file to use as title
 * @param {string} filePath - Path to the markdown file
 * @returns {string|null} - The extracted title or null
 */
function extractTitleFromMarkdown(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    const h1Match = content.match(/^#\s+(.+)$/m);
    if (h1Match) {
      return h1Match[1].trim().replace(/`([^`]+)`/g, '$1'); // Remove backticks
    }
  } catch (error) {
    console.error(`Error reading ${filePath}:`, error.message);
  }
  return null;
}

/**
 * Generate a human-readable label from a filename
 * @param {string} filename - The filename (without extension)
 * @returns {string} - Human-readable label
 */
function generateLabel(filename) {
  return filename
    .split(/[-_]/)
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

/**
 * Generate a category label from directory name
 * @param {string} dirname - Directory name
 * @returns {string} - Human-readable category label
 */
function generateCategoryLabel(dirname) {
  return dirname.charAt(0).toUpperCase() + dirname.slice(1);
}

/**
 * Recursively scan directory for markdown files
 * @param {string} dir - Directory to scan
 * @param {string} baseDir - Base directory for relative paths
 * @param {string} category - Current category
 * @returns {Array} - Array of document entries
 */
function scanDirectory(dir, baseDir = dir, category = 'root') {
  const entries = [];
  
  try {
    const items = fs.readdirSync(dir);
    
    for (const item of items) {
      const fullPath = path.join(dir, item);
      const stat = fs.statSync(fullPath);
      
      if (stat.isDirectory()) {
        // Recursively scan subdirectories
        const subCategory = item;
        const subEntries = scanDirectory(fullPath, baseDir, subCategory);
        entries.push(...subEntries);
      } else if (stat.isFile() && item.endsWith('.md')) {
        // Process markdown file
        const relativePath = path.relative(baseDir, fullPath);
        const pathWithoutExt = relativePath.replace(/\.md$/, '').replace(/\\/g, '/');
        const filename = path.basename(item, '.md');
        
        // Extract title from markdown or generate from filename
        let title = extractTitleFromMarkdown(fullPath);
        if (!title) {
          title = generateLabel(filename);
        }
        
        // Special handling for Overview.md files
        if (filename.toLowerCase() === 'overview') {
          title = `${generateCategoryLabel(category)} Overview`;
        }
        
        // Special handling for README.md files
        if (filename.toLowerCase() === 'readme') {
          if (category === 'root') {
            title = 'Getting Started';
          } else {
            title = `${generateCategoryLabel(category)} Overview`;
          }
        }
        
        entries.push({
          title,
          path: pathWithoutExt,
          displayPath: `/docs/${pathWithoutExt}`,
          category,
          filename: item
        });
      }
    }
  } catch (error) {
    console.error(`Error scanning directory ${dir}:`, error.message);
  }
  
  return entries;
}

/**
 * Copy markdown files from docs/ to public/docs/
 * @param {string} sourceDir - Source directory
 * @param {string} destDir - Destination directory
 */
function copyMarkdownFiles(sourceDir, destDir) {
  try {
    const items = fs.readdirSync(sourceDir);
    
    for (const item of items) {
      const sourcePath = path.join(sourceDir, item);
      const destPath = path.join(destDir, item);
      const stat = fs.statSync(sourcePath);
      
      if (stat.isDirectory()) {
        // Create destination directory if it doesn't exist
        if (!fs.existsSync(destPath)) {
          fs.mkdirSync(destPath, { recursive: true });
        }
        // Recursively copy subdirectory
        copyMarkdownFiles(sourcePath, destPath);
      } else if (stat.isFile() && item.endsWith('.md')) {
        // Copy markdown file
        fs.copyFileSync(sourcePath, destPath);
        console.log(`  Copied: ${path.relative(DOCS_SOURCE_DIR, sourcePath)}`);
      }
    }
  } catch (error) {
    console.error(`Error copying files from ${sourceDir}:`, error.message);
  }
}

/**
 * Sort entries by category and title for consistent ordering
 * @param {Array} entries - Array of document entries
 * @returns {Array} - Sorted entries
 */
function sortEntries(entries) {
  const categoryOrder = ['root', 'backend', 'frontend', 'server'];
  
  return entries.sort((a, b) => {
    // First sort by category
    const categoryA = categoryOrder.indexOf(a.category);
    const categoryB = categoryOrder.indexOf(b.category);
    
    if (categoryA !== categoryB) {
      return categoryA - categoryB;
    }
    
    // Within same category, Overview/README comes first
    const filenameA = a.filename.toLowerCase();
    const filenameB = b.filename.toLowerCase();
    
    if ((filenameA === 'overview.md' || filenameA === 'readme.md') && 
        (filenameB !== 'overview.md' && filenameB !== 'readme.md')) {
      return -1;
    }
    if ((filenameB === 'overview.md' || filenameB === 'readme.md') && 
        (filenameA !== 'overview.md' && filenameA !== 'readme.md')) {
      return 1;
    }
    
    // Then sort alphabetically by title
    return a.title.localeCompare(b.title);
  });
}

/**
 * Main function to generate the manifest
 */
function generateManifest() {
  console.log('🔍 Scanning documentation files...\n');
  
  // Ensure public/docs directory exists
  if (!fs.existsSync(DOCS_PUBLIC_DIR)) {
    fs.mkdirSync(DOCS_PUBLIC_DIR, { recursive: true });
    console.log(`✅ Created directory: ${DOCS_PUBLIC_DIR}\n`);
  }
  
  // Check if source docs directory exists
  if (!fs.existsSync(DOCS_SOURCE_DIR)) {
    console.error(`❌ Error: Source docs directory not found: ${DOCS_SOURCE_DIR}`);
    return;
  }
  
  console.log('📄 Copying markdown files to public/docs/...\n');
  copyMarkdownFiles(DOCS_SOURCE_DIR, DOCS_PUBLIC_DIR);
  
  console.log('\n📊 Generating manifest...\n');
  
  // Scan for markdown files
  let entries = scanDirectory(DOCS_SOURCE_DIR);
  
  // Sort entries
  entries = sortEntries(entries);
  
  // Remove the filename property before writing to manifest
  const manifestEntries = entries.map(entry => {
    const { filename, ...rest } = entry;
    return rest;
  });
  
  // Write manifest
  const manifestContent = JSON.stringify(manifestEntries, null, 2);
  fs.writeFileSync(MANIFEST_PATH, manifestContent);
  
  console.log(`✅ Generated manifest with ${manifestEntries.length} entries:`);
  console.log(`   ${MANIFEST_PATH}\n`);
  
  // Display summary
  console.log('📋 Manifest Summary:\n');
  const byCategory = {};
  manifestEntries.forEach(entry => {
    if (!byCategory[entry.category]) {
      byCategory[entry.category] = [];
    }
    byCategory[entry.category].push(entry.title);
  });
  
  Object.entries(byCategory).forEach(([category, titles]) => {
    const categoryLabel = category === 'root' ? 'Root Level' : generateCategoryLabel(category);
    console.log(`  ${categoryLabel}:`);
    titles.forEach(title => {
      console.log(`    - ${title}`);
    });
    console.log('');
  });
  
  console.log('✨ Documentation manifest generated successfully!\n');
  console.log('💡 Next steps:');
  console.log('   1. Review the generated manifest.json');
  console.log('   2. Documentation will automatically appear in the sidebar');
  console.log('   3. Search will index all new documents\n');
}

// Run the generator
try {
  generateManifest();
} catch (error) {
  console.error('❌ Error generating manifest:', error);
}
