// src/constants/badgeConstants.js

import { cva } from "class-variance-authority";

/**
 * Badge Component - Highly customizable badge for status, media types, and more
 */

// Badge variants configuration
export const badgeVariants = cva(
  "inline-flex items-center font-medium border transition-colors",
  {
    variants: {
      variant: {
        default: "bg-gray-500/10 text-gray-600 dark:text-gray-400 border-gray-500/20",
        primary: "bg-primary-main/10 text-primary-main dark:bg-primary-main/20 border-primary-main/20",
        success: "bg-green-500/10 text-green-600 dark:text-green-400 border-green-500/20",
        warning: "bg-yellow-500/10 text-yellow-600 dark:text-yellow-400 border-yellow-500/20",
        danger: "bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/20",
        info: "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20",
        purple: "bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-500/20",
        indigo: "bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 border-indigo-500/20",
        pink: "bg-pink-500/10 text-pink-600 dark:text-pink-400 border-pink-500/20",
      },
      size: {
        sm: "px-2 py-0.5 text-xs gap-1",
        md: "px-2.5 py-1 text-xs gap-1.5",
        lg: "px-3 py-1.5 text-sm gap-2",
      },
      rounded: {
        default: "rounded",
        full: "rounded-full",
        lg: "rounded-lg",
      },
      animate: {
        none: "",
        pulse: "animate-pulse",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "md",
      rounded: "full",
      animate: "none",
    },
  }
);

// Predefined status configurations
export const statusConfigs = {
  ANALYZED: {
    variant: "success",
    label: "analyzed",
  },
  PARTIALLY_ANALYZED: {
    variant: "info",
    label: "partially analyzed",
  },
  PROCESSING: {
    variant: "warning",
    label: "processing",
    animate: "pulse",
  },
  QUEUED: {
    variant: "indigo",
    label: "queued",
  },
  FAILED: {
    variant: "danger",
    label: "failed",
  },
  COMPLETED: {
    variant: "success",
    label: "completed",
  },
  PENDING: {
    variant: "warning",
    label: "pending",
  },
};

// Predefined media type configurations
export const mediaTypeConfigs = {
  VIDEO: {
    variant: "purple",
    label: "Video",
  },
  AUDIO: {
    variant: "info",
    label: "Audio",
  },
  IMAGE: {
    variant: "success",
    label: "Image",
  },
};

// Predefined device configurations
export const deviceConfigs = {
  CUDA: {
    variant: "success",
    label: "CUDA",
  },
  CPU: {
    variant: "default",
    label: "CPU",
  },
  MPS: {
    variant: "purple",
    label: "MPS",
  },
};