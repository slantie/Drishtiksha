// src/constants/buttonConstants.js

import { cva } from "class-variance-authority";

// REFACTOR: Variants are now explicitly designed for a Primary/Secondary UI pattern.
// Consistent focus rings and transitions have been applied to all variants for a polished feel.
export const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap text-sm font-semibold transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-primary-main disabled:pointer-events-none disabled:opacity-60",
  {
    variants: {
      variant: {
        // Primary Action Button: Use for the main call-to-action on a page.
        default:
          "bg-primary-main text-white shadow-sm hover:bg-primary-main/90 rounded-full",
        // Destructive Action Button: Use for actions that delete data or are irreversible.
        destructive:
          "bg-red-600/25 text-red-500 shadow-sm hover:bg-red-600/40 rounded-full",
        // Secondary Action Button: Use for the less important action in a pair.
        outline:
          "border border-light-secondary dark:border-dark-secondary bg-transparent hover:bg-light-hover dark:hover:bg-dark-hover rounded-full",
        // Alternative Secondary Button: A subtle alternative to the outline style.
        secondary:
          "bg-light-muted-background dark:bg-dark-secondary text-light-text dark:text-dark-text hover:bg-light-hover dark:hover:bg-dark-hover rounded-full",
        // Ghost Button: Use for low-emphasis actions, often within other components like tables.
        ghost: "hover:bg-light-hover dark:hover:bg-dark-hover rounded-full",
        // Link Button: Use for actions that should look like a hyperlink.
        link: "text-primary-main underline-offset-4 hover:underline rounded-full",
      },
      size: {
        default: "h-10 px-4 py-2 rounded-full",
        sm: "h-9 px-3 rounded-full ",
        lg: "h-11 px-8 rounded-full",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);