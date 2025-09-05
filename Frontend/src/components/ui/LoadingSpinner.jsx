// src/components/ui/LoadingSpinner.jsx

import React from "react";
import { Loader2 } from "lucide-react";
import { cn } from "../../lib/utils";

const sizeClasses = {
  sm: "h-4 w-4",
  md: "h-6 w-6",
  lg: "h-10 w-10",
  xl: "h-16 w-16", // Added an extra large size
};

/**
 * A consistent, themeable loading spinner.
 * @param {object} props - The component props.
 * @param {'sm'|'md'|'lg'|'xl'} [props.size='md'] - The size of the spinner.
 * @param {string} [props.text] - Optional text to display below the spinner.
 * @param {string} [props.className] - Additional CSS classes for the container.
 * @param {string} [props.spinnerClassName] - Additional CSS classes for the spinner icon itself.
 */
export const LoadingSpinner = ({
  size = "md",
  text,
  className,
  spinnerClassName,
}) => (
  <div
    className={cn("flex flex-col items-center justify-center gap-2", className)}
  >
    <Loader2
      className={cn(
        "animate-spin text-primary-main",
        sizeClasses[size],
        spinnerClassName
      )}
      aria-label="Loading" // Accessibility
      role="status" // Accessibility
    />
    {text && (
      <p className="text-sm text-light-muted-text dark:text-dark-muted-text">
        {text}
      </p>
    )}
  </div>
);

/**
 * A full-page loader for initial page loads or heavy operations.
 * @param {object} props - The component props.
 * @param {string} [props.text='Loading...'] - Optional text to display below the spinner.
 */
export const PageLoader = ({ text = "Loading..." }) => (
  <div className="flex h-screen w-full items-center justify-center bg-light-background dark:bg-dark-background">
    {" "}
    {/* Ensure full-page background */}
    <LoadingSpinner size="lg" text={text} />
  </div>
);
