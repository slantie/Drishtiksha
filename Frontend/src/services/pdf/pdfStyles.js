// src/services/pdf/pdfStyles.js
// Centralized design system for PDF reports

/**
 * Color Palette for PDF Reports
 * All colors are hex codes for consistent rendering across PDF viewers
 */
export const PDFColors = {
  // Brand Colors
  primary: '#8155c6',
  primaryLight: '#9d7dd4',
  primaryDark: '#6541a5',
  
  // Status Colors
  authentic: '#16a34a',
  authenticLight: '#22c55e',
  authenticDark: '#15803d',
  deepfake: '#dc2626',
  deepfakeLight: '#ef4444',
  deepfakeDark: '#b91c1c',
  warning: '#f59e0b',
  warningLight: '#fbbf24',
  failed: '#9ca3af',
  success: '#10b981',
  
  // Grayscale
  black: '#1B1B1F',
  gray900: '#27272a',
  gray800: '#3C3C43',
  gray700: '#52525b',
  gray600: '#67676C',
  gray500: '#8B8B90',
  gray400: '#ABABAF',
  gray300: '#d4d4d8',
  gray200: '#EBEBEF',
  gray100: '#f9fafb',
  gray50: '#fafafa',
  white: '#FFFFFF',
  
  // Chart Colors (for multi-line visualizations)
  chart: {
    blue: '#3b82f6',
    green: '#10b981',
    purple: '#8b5cf6',
    orange: '#f97316',
    pink: '#ec4899',
    teal: '#14b8a6',
    indigo: '#6366f1',
    yellow: '#eab308',
    cyan: '#06b6d4',
    rose: '#f43f5e',
  },
  
  // Background gradients
  gradients: {
    primarySubtle: ['#f5f3ff', '#ede9fe'],
    authenticSubtle: ['#f0fdf4', '#dcfce7'],
    deepfakeSubtle: ['#fef2f2', '#fee2e2'],
  }
};

/**
 * Typography System
 * Font sizes in points, line heights as multipliers
 */
export const PDFTypography = {
  // Font Families (react-pdf supported fonts)
  fontFamily: {
    sans: 'Helvetica',
    sansAlt: 'Helvetica-Bold',
    mono: 'Courier',
  },
  
  // Font Sizes (in points)
  fontSize: {
    h1: 24,
    h2: 18,
    h3: 14,
    h4: 12,
    body: 10,
    small: 9,
    tiny: 8,
    micro: 7,
  },
  
  // Font Weights (react-pdf uses font names for weights)
  fontWeight: {
    normal: 'normal',
    bold: 'bold',
  },
  
  // Line Heights (as multipliers)
  lineHeight: {
    tight: 1.2,
    snug: 1.3,
    normal: 1.5,
    relaxed: 1.75,
    loose: 2,
  },
  
  // Letter Spacing (in points)
  letterSpacing: {
    tighter: -0.5,
    tight: -0.25,
    normal: 0,
    wide: 0.25,
    wider: 0.5,
  }
};

/**
 * Spacing System
 * All values in points (pt)
 */
export const PDFSpacing = {
  // Base spacing units
  0: 0,
  1: 2,
  2: 4,
  3: 6,
  4: 8,
  5: 10,
  6: 12,
  8: 16,
  10: 20,
  12: 24,
  16: 32,
  20: 40,
  24: 48,
  32: 64,
  
  // Named spacing (for semantic use)
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
  '2xl': 32,
  '3xl': 48,
  '4xl': 64,
  
  // Page-specific spacing
  page: {
    margin: 48,      // 48pt = ~16mm
    marginTop: 48,
    marginBottom: 48,
    marginLeft: 48,
    marginRight: 48,
  },
  
  // Section spacing
  section: {
    marginBottom: 24,
    titleMarginBottom: 12,
  },
};

/**
 * Border Styles
 */
export const PDFBorders = {
  width: {
    thin: 0.5,
    normal: 1,
    thick: 2,
    thicker: 3,
  },
  
  radius: {
    none: 0,
    sm: 2,
    md: 4,
    lg: 6,
    xl: 8,
    full: 999,
  },
  
  style: {
    solid: 'solid',
    dashed: 'dashed',
    dotted: 'dotted',
  }
};

/**
 * Shadow Definitions (for card-like elements)
 * Note: react-pdf doesn't support box-shadow, but we can simulate with borders
 */
export const PDFShadows = {
  none: {},
  sm: {
    borderWidth: PDFBorders.width.thin,
    borderColor: PDFColors.gray200,
  },
  md: {
    borderWidth: PDFBorders.width.normal,
    borderColor: PDFColors.gray300,
  },
  lg: {
    borderWidth: PDFBorders.width.thick,
    borderColor: PDFColors.gray300,
  }
};

/**
 * Layout Presets
 * Common layout patterns for consistent design
 */
export const PDFLayouts = {
  // A4 Page dimensions (in points: 1 inch = 72 points)
  a4: {
    width: 595.28,   // 210mm
    height: 841.89,  // 297mm
  },
  
  // Content area (A4 minus margins)
  contentArea: {
    width: 499.28,   // A4 width - (2 * 48pt margins)
    height: 745.89,  // A4 height - (2 * 48pt margins)
  },
  
  // Grid systems
  grid: {
    cols2: {
      columnGap: 16,
      columnWidth: 241.64, // (contentArea.width - gap) / 2
    },
    cols3: {
      columnGap: 16,
      columnWidth: 155.76, // (contentArea.width - 2*gap) / 3
    },
    cols4: {
      columnGap: 12,
      columnWidth: 113.32, // (contentArea.width - 3*gap) / 4
    }
  },
  
  // Flex layouts
  flex: {
    row: {
      flexDirection: 'row',
    },
    column: {
      flexDirection: 'column',
    },
    spaceBetween: {
      justifyContent: 'space-between',
    },
    center: {
      justifyContent: 'center',
      alignItems: 'center',
    }
  }
};

/**
 * Common Style Compositions
 * Reusable style objects for common UI patterns
 */
export const PDFCommonStyles = {
  // Page title with underline
  pageTitle: {
    fontSize: PDFTypography.fontSize.h2,
    fontFamily: PDFTypography.fontFamily.sansAlt,
    color: PDFColors.primary,
    marginBottom: PDFSpacing.md,
    paddingBottom: PDFSpacing.sm,
    borderBottomWidth: PDFBorders.width.thick,
    borderBottomColor: PDFColors.primary,
    borderBottomStyle: PDFBorders.style.solid,
  },
  
  // Section title
  sectionTitle: {
    fontSize: PDFTypography.fontSize.h3,
    fontFamily: PDFTypography.fontFamily.sansAlt,
    color: PDFColors.gray900,
    marginBottom: PDFSpacing.sm,
    marginTop: PDFSpacing.lg,
  },
  
  // Card container
  card: {
    backgroundColor: PDFColors.white,
    borderWidth: PDFBorders.width.normal,
    borderColor: PDFColors.gray200,
    borderRadius: PDFBorders.radius.md,
    borderStyle: PDFBorders.style.solid,
    padding: PDFSpacing.md,
  },
  
  // Badge base
  badge: {
    paddingVertical: PDFSpacing[2],
    paddingHorizontal: PDFSpacing.sm,
    borderRadius: PDFBorders.radius.full,
    fontSize: PDFTypography.fontSize.tiny,
    fontFamily: PDFTypography.fontFamily.sansAlt,
  },
  
  // Table cell
  tableCell: {
    paddingVertical: PDFSpacing.sm,
    paddingHorizontal: PDFSpacing.xs,
    borderBottomWidth: PDFBorders.width.thin,
    borderBottomColor: PDFColors.gray200,
    borderBottomStyle: PDFBorders.style.solid,
  },
  
  // Confidence bar
  confidenceBar: {
    height: 20,
    backgroundColor: PDFColors.gray100,
    borderRadius: PDFBorders.radius.sm,
    overflow: 'hidden',
  },
  
  // Footer text
  footerText: {
    fontSize: PDFTypography.fontSize.tiny,
    color: PDFColors.gray600,
    textAlign: 'center',
  }
};

/**
 * Icon Replacements
 * Since we can't use lucide-react in PDFs, we use Unicode symbols
 */
export const PDFIcons = {
  // Status
  checkmark: '✓',
  cross: '✗',
  warning: '⚠',
  info: 'ℹ',
  
  // Predictions
  authentic: '✓',
  deepfake: '⚠',
  failed: '✗',
  
  // Sections
  summary: '📊',
  details: '🔍',
  chart: '📈',
  document: '📄',
  clock: '⏱',
  shield: '🛡',
  brain: '🧠',
  
  // Arrows
  arrowRight: '→',
  arrowLeft: '←',
  arrowUp: '↑',
  arrowDown: '↓',
  
  // Other
  bullet: '•',
  star: '★',
  circle: '○',
  circleFilled: '●',
};

/**
 * Helper function to get confidence color
 */
export const getConfidenceColor = (confidence, prediction = 'REAL') => {
  if (prediction === 'FAKE') {
    return confidence >= 0.9 ? PDFColors.deepfakeDark :
           confidence >= 0.75 ? PDFColors.deepfake :
           confidence >= 0.6 ? PDFColors.deepfakeLight :
           PDFColors.warning;
  } else {
    return confidence >= 0.9 ? PDFColors.authenticDark :
           confidence >= 0.75 ? PDFColors.authentic :
           confidence >= 0.6 ? PDFColors.authenticLight :
           PDFColors.warning;
  }
};

/**
 * Helper function to get status color
 */
export const getStatusColor = (status) => {
  const colors = {
    'COMPLETED': PDFColors.success,
    'ANALYZED': PDFColors.success,
    'FAILED': PDFColors.deepfake,
    'PROCESSING': PDFColors.warning,
    'QUEUED': PDFColors.gray500,
    'PENDING': PDFColors.gray500,
  };
  return colors[status] || PDFColors.gray500;
};

/**
 * Helper function to format confidence as percentage
 */
export const formatConfidencePercent = (confidence) => {
  return `${(confidence * 100).toFixed(1)}%`;
};
