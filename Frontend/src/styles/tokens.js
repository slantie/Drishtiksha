/**
 * Design System Tokens
 * Based on tailwind.config.js theme
 * 
 * Use these tokens for consistent spacing, typography, and colors
 * across all components.
 */

// ============================================
// SPACING TOKENS
// ============================================
export const spacing = {
  xs: 'gap-2 space-y-2',   // 8px  - Tight spacing (chips, tags, inline elements)
  sm: 'gap-3 space-y-3',   // 12px - Compact spacing (form fields, list items)
  md: 'gap-4 space-y-4',   // 16px - Default spacing (cards, sections)
  lg: 'gap-6 space-y-6',   // 24px - Generous spacing (major sections)
  xl: 'gap-8 space-y-8',   // 32px - Spacious (page-level sections)
};

// For padding specifically
export const padding = {
  xs: 'p-2',   // 8px
  sm: 'p-3',   // 12px
  md: 'p-4',   // 16px
  lg: 'p-6',   // 24px
  xl: 'p-8',   // 32px
};

// ============================================
// TYPOGRAPHY TOKENS
// ============================================
export const typography = {
  // Headings
  h1: 'text-4xl font-bold leading-tight',        // 36px - Page titles
  h2: 'text-3xl font-bold leading-tight',        // 30px - Section titles
  h3: 'text-2xl font-semibold leading-snug',     // 24px - Card titles
  h4: 'text-xl font-semibold leading-snug',      // 20px - Subsections
  
  // Body text
  body: 'text-base leading-normal',              // 16px - Normal text
  bodyLarge: 'text-lg leading-relaxed',          // 18px - Emphasis text
  small: 'text-sm leading-relaxed',              // 14px - Captions, meta info
  tiny: 'text-xs leading-relaxed',               // 12px - Labels, timestamps
  
  // Special
  lead: 'text-xl leading-relaxed',               // 20px - Hero/intro text
};

// ============================================
// COLOR TOKENS (Using theme colors)
// ============================================
export const colors = {
  // Semantic status colors
  status: {
    success: 'bg-green-500 text-white',
    error: 'bg-red-500 text-white',
    warning: 'bg-yellow-500 text-white',
    info: 'bg-blue-500 text-white',
    processing: 'bg-primary-main text-white',
    queued: 'bg-light-tertiary dark:bg-dark-tertiary text-white',
  },
  
  // Background colors (using theme)
  background: {
    page: 'bg-light-muted-background dark:bg-dark-background',
    card: 'bg-light-background dark:bg-dark-muted-background',
    hover: 'hover:bg-light-hover dark:hover:bg-dark-hover',
    secondary: 'bg-light-secondary dark:bg-dark-secondary',
  },
  
  // Text colors
  text: {
    primary: 'text-light-text dark:text-dark-text',
    muted: 'text-light-muted-text dark:text-dark-muted-text',
    noisy: 'text-light-noisy-text dark:text-dark-noisy-text',
  },
  
  // Accent colors
  accent: {
    primary: 'text-primary-main',
    primaryBg: 'bg-primary-main',
    highlight: 'text-light-highlight dark:text-dark-highlight',
  },
  
  // Borders
  border: {
    default: 'border-light-secondary dark:border-dark-secondary',
    muted: 'border-light-noisy-background dark:border-dark-noisy-background',
  },
};

// ============================================
// COMPONENT TOKENS
// ============================================

// Button sizes
export const button = {
  xs: 'px-2 py-1 text-xs',           // Extra small
  sm: 'px-3 py-1.5 text-sm',         // Small
  md: 'px-4 py-2 text-base',         // Default
  lg: 'px-6 py-3 text-lg',           // Large
  xl: 'px-8 py-4 text-xl',           // Extra large
  
  // Icon button sizes (square)
  icon: {
    xs: 'p-1',      // 8px padding
    sm: 'p-2',      // 12px padding
    md: 'p-2.5',    // 14px padding
    lg: 'p-3',      // 16px padding
  },
};

// Icon sizes (for Lucide icons)
export const icon = {
  xs: 'h-3 w-3',   // 12px
  sm: 'h-4 w-4',   // 16px
  md: 'h-5 w-5',   // 20px
  lg: 'h-6 w-6',   // 24px
  xl: 'h-8 w-8',   // 32px
};

// Card/Container styling
export const card = {
  base: `${colors.background.card} rounded-lg shadow-sm ${colors.border.default} border`,
  padded: `${colors.background.card} rounded-lg shadow-sm ${colors.border.default} border ${padding.md}`,
  hover: `${colors.background.card} rounded-lg shadow-sm ${colors.border.default} border ${colors.background.hover} transition-colors`,
};

// ============================================
// RESPONSIVE BREAKPOINTS
// ============================================
// For 11" screens and above
export const breakpoints = {
  // 11" laptop: typically 1366x768
  sm: 'sm:',    // 640px  - Small tablets
  md: 'md:',    // 768px  - Tablets
  lg: 'lg:',    // 1024px - Small laptops
  xl: 'xl:',    // 1280px - Standard laptops (11" to 13")
  '2xl': '2xl:', // 1536px - Large screens (15"+)
};

// Grid column patterns
export const grid = {
  // Responsive grid for cards/items
  responsive: 'grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4',
  
  // Two column layouts (sidebar + main)
  twoColumn: 'grid grid-cols-1 xl:grid-cols-3',
  
  // Stats/metrics grid
  stats: 'grid grid-cols-2 lg:grid-cols-4 gap-4',
};

// ============================================
// ANIMATION/TRANSITION TOKENS
// ============================================
export const transition = {
  fast: 'transition-all duration-150 ease-in-out',
  normal: 'transition-all duration-300 ease-in-out',
  slow: 'transition-all duration-500 ease-in-out',
};

// ============================================
// SHADOW TOKENS
// ============================================
export const shadow = {
  sm: 'shadow-sm',
  md: 'shadow-md',
  lg: 'shadow-lg',
  xl: 'shadow-xl',
  none: 'shadow-none',
};

// ============================================
// USAGE EXAMPLES
// ============================================

// Example component using tokens:
/*
import { spacing, typography, colors, button, card } from '@/styles/tokens';

function MyComponent() {
  return (
    <div className={`${card.padded} ${spacing.md}`}>
      <h2 className={`${typography.h2} ${colors.text.primary}`}>
        Title
      </h2>
      <p className={`${typography.body} ${colors.text.muted}`}>
        Description text
      </p>
      <button className={`${button.md} ${colors.accent.primaryBg}`}>
        Action
      </button>
    </div>
  );
}
*/

export default {
  spacing,
  padding,
  typography,
  colors,
  button,
  icon,
  card,
  breakpoints,
  grid,
  transition,
  shadow,
};
