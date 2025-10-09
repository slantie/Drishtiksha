// src/components/ui/Card.jsx

import * as React from "react";
import { cn } from "../../lib/utils";

// REFACTOR: The base Card component is refined for consistent styling.
// It uses theme-compliant background colors and borders for both light and dark modes.
const Card = React.forwardRef(({ className, ...props }, ref) => (
    <div
        ref={ref}
        className={cn(
            "rounded-xl border border-light-secondary dark:border-dark-secondary bg-light-background dark:bg-dark-muted-background shadow-sm",
            className
        )}
        {...props}
    />
));
Card.displayName = "Card";

// REFACTOR: Standardized padding and added a bottom border for clear separation of the header.
const CardHeader = React.forwardRef(({ className, ...props }, ref) => (
    <div
        ref={ref}
        className={cn(
            "flex flex-col space-y-1.5 p-4 border-b border-light-secondary dark:border-dark-secondary",
            className
        )}
        {...props}
    />
));
CardHeader.displayName = "CardHeader";

// REFACTOR: Typography is now more distinct for titles, improving visual hierarchy.
const CardTitle = React.forwardRef(({ className, ...props }, ref) => (
    <h3
        ref={ref}
        className={cn(
            "text-lg font-semibold leading-none tracking-tight",
            className
        )}
        {...props}
    />
));
CardTitle.displayName = "CardTitle";

// REFACTOR: Ensured description text uses the correct muted text color from the theme.
const CardDescription = React.forwardRef(({ className, ...props }, ref) => (
    <p
        ref={ref}
        className={cn(
            "text-sm text-light-muted-text dark:text-dark-muted-text",
            className
        )}
        {...props}
    />
));
CardDescription.displayName = "CardDescription";

// REFACTOR: Standardized content padding.
const CardContent = React.forwardRef(({ className, ...props }, ref) => (
    <div ref={ref} className={cn("p-6", className)} {...props} />
));
CardContent.displayName = "CardContent";

// REFACTOR: Added a top border to clearly separate the footer from the content.
const CardFooter = React.forwardRef(({ className, ...props }, ref) => (
    <div
        ref={ref}
        className={cn(
            "flex items-center p-6 border-t border-light-secondary dark:border-dark-secondary",
            className
        )}
        {...props}
    />
));
CardFooter.displayName = "CardFooter";

export {
    Card,
    CardHeader,
    CardFooter,
    CardTitle,
    CardDescription,
    CardContent,
};
