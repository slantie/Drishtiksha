// src/utils/modelColors.js
// Centralized color management for model visualization across charts

// Extended color palette for models (15 distinct colors)
const MODEL_COLOR_PALETTE = [
  "#3b82f6", // blue
  "#ef4444", // red
  "#10b981", // green
  "#f59e0b", // amber
  "#8b5cf6", // purple
  "#ec4899", // pink
  "#14b8a6", // teal
  "#f97316", // orange
  "#84cc16", // lime
  "#06b6d4", // cyan
  "#6366f1", // indigo
  "#a855f7", // violet
  "#22c55e", // green-500
  "#eab308", // yellow
  "#f43f5e", // rose
];

// Cache for model name to color mapping
const modelColorCache = new Map();

/**
 * Get a consistent color for a model across all charts
 * @param {string} modelName - The name of the model
 * @param {Array<string>} allModelNames - Optional: array of all model names to ensure consistent indexing
 * @returns {string} Hex color code
 */
export const getModelColor = (modelName, allModelNames = null) => {
  // If we have a predefined list, use its index for consistency
  if (allModelNames && Array.isArray(allModelNames)) {
    const index = allModelNames.indexOf(modelName);
    if (index !== -1) {
      return MODEL_COLOR_PALETTE[index % MODEL_COLOR_PALETTE.length];
    }
  }

  // Check cache first
  if (modelColorCache.has(modelName)) {
    return modelColorCache.get(modelName);
  }

  // Generate a consistent color based on model name hash
  let hash = 0;
  for (let i = 0; i < modelName.length; i++) {
    hash = modelName.charCodeAt(i) + ((hash << 5) - hash);
  }
  const index = Math.abs(hash) % MODEL_COLOR_PALETTE.length;
  const color = MODEL_COLOR_PALETTE[index];

  // Cache the result
  modelColorCache.set(modelName, color);

  return color;
};

/**
 * Get colors for multiple models, ensuring consistency
 * @param {Array<string>} modelNames - Array of model names
 * @returns {Object} Map of model name to color
 */
export const getModelColorMap = (modelNames) => {
  const colorMap = {};
  modelNames.forEach((modelName) => {
    colorMap[modelName] = getModelColor(modelName, modelNames);
  });
  return colorMap;
};

/**
 * Clear the color cache (useful if you want to reset color assignments)
 */
export const clearModelColorCache = () => {
  modelColorCache.clear();
};

/**
 * Get a lighter version of a color for backgrounds
 * @param {string} hexColor - Hex color code
 * @param {number} opacity - Opacity value (0-1)
 * @returns {string} RGBA color string
 */
export const getModelColorWithOpacity = (hexColor, opacity = 0.1) => {
  const r = parseInt(hexColor.slice(1, 3), 16);
  const g = parseInt(hexColor.slice(3, 5), 16);
  const b = parseInt(hexColor.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${opacity})`;
};

export { MODEL_COLOR_PALETTE };
