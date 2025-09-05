// src/components/ui/Alert.jsx

import * as React from "react";
import { cva } from "class-variance-authority";
import { cn } from "../../lib/utils";

// NEW COMPONENT: A versatile alert component for consistent user feedback.
const alertVariants = cva(
  "relative w-full rounded-lg border p-4 [&>svg~*]:pl-7 [&>svg+div]:translate-y-[-3px] [&>svg]:absolute [&>svg]:left-4 [&>svg]:top-4", // Removed [&>svg]:text-foreground
  {
    variants: {
      variant: {
        default:
          "bg-light-background dark:bg-dark-muted-background text-light-text dark:text-dark-text border-light-secondary dark:border-dark-secondary [&>svg]:text-light-text dark:[&>svg]:text-dark-text", // Explicitly define colors
        destructive:
          "border-red-500/50 text-red-500 dark:border-red-500 [&>svg]:text-red-500 bg-red-100/50 dark:bg-red-900/20", // Added background color
        success:
          "border-green-500/50 text-green-600 dark:border-green-500 [&>svg]:text-green-600 bg-green-100/50 dark:bg-green-900/20", // Added background color
        // New info variant
        info: "border-blue-500/50 text-blue-600 dark:border-blue-500 [&>svg]:text-blue-600 bg-blue-100/50 dark:bg-blue-900/20",
        // New warning variant
        warning:
          "border-yellow-500/50 text-yellow-600 dark:border-yellow-500 [&>svg]:text-yellow-600 bg-yellow-100/50 dark:bg-yellow-900/20",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

const Alert = React.forwardRef(({ className, variant, ...props }, ref) => (
  <div
    ref={ref}
    role="alert"
    className={cn(alertVariants({ variant }), className)}
    {...props}
  />
));
Alert.displayName = "Alert";

const AlertTitle = React.forwardRef(({ className, ...props }, ref) => (
  <h5
    ref={ref}
    className={cn("mb-1 font-medium leading-none tracking-tight", className)}
    {...props}
  />
));
AlertTitle.displayName = "AlertTitle";

const AlertDescription = React.forwardRef(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("text-sm [&_p]:leading-relaxed", className)}
    {...props}
  />
));
AlertDescription.displayName = "AlertDescription";

export { Alert, AlertTitle, AlertDescription };
