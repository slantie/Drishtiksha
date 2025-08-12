/**
 * @file src/components/ui/Card.jsx
 * @description A flexible card component with multiple variants and compound components.
 */

"use client";

import React from "react";
import PropTypes from "prop-types";
import { cva } from "class-variance-authority";
import { cn } from "../../lib/utils";

// CVA definition remains the same
const cardVariants = cva("rounded-lg border transition-colors", {
    variants: {
        variant: {
            default: "bg-light-background border-light-secondary dark:bg-dark-background dark:border-dark-secondary",
            elevated: "bg-light-background border-light-secondary shadow-md dark:bg-dark-background dark:border-dark-secondary dark:shadow-lg",
            outlined: "bg-transparent border-2 border-light-secondary dark:border-dark-secondary",
            ghost: "bg-transparent border-transparent hover:bg-light-hover dark:hover:bg-dark-hover",
            gradient: "bg-gradient-to-br from-primary-lighter to-light-background border-primary-light dark:from-dark-background dark:to-dark-secondary dark:border-dark-secondary", // Adjusted dark gradient
        },
        padding: {
            none: "p-0",
            sm: "p-3",
            default: "p-4",
            lg: "p-6",
            xl: "p-8",
        },
    },
    defaultVariants: {
        variant: "default",
        padding: "default",
    },
});

// --- Main Card Component ---

function Card({ className, variant, padding, ...props }) {
    return (
        <div
            className={cn(cardVariants({ variant, padding }), className)}
            {...props}
        />
    );
}

// --- Card Sub-components ---

const CardHeader = React.forwardRef(({ className, ...props }, ref) => (
    <div
        ref={ref}
        className={cn("flex flex-col space-y-1.5 p-6", className)}
        {...props}
    />
));
CardHeader.displayName = "CardHeader";

const CardTitle = React.forwardRef(({ className, ...props }, ref) => (
    <h3
        ref={ref}
        className={cn(
            "text-xl font-semibold leading-none tracking-tight text-light-text dark:text-dark-text", // Adjusted colors to match theme
            className
        )}
        {...props}
    />
));
CardTitle.displayName = "CardTitle";

const CardDescription = React.forwardRef(({ className, ...props }, ref) => (
    <p
        ref={ref}
        className={cn(
            "text-sm text-light-muted-text dark:text-dark-noisy-text", // Adjusted colors to match theme
            className
        )}
        {...props}
    />
));
CardDescription.displayName = "CardDescription";

const CardContent = React.forwardRef(({ className, ...props }, ref) => (
    <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
));
CardContent.displayName = "CardContent";

const CardFooter = React.forwardRef(({ className, ...props }, ref) => (
    <div
        ref={ref}
        className={cn("flex items-center p-6 pt-0", className)}
        {...props}
    />
));
CardFooter.displayName = "CardFooter";

// --- PropTypes Definitions ---

Card.propTypes = {
    className: PropTypes.string,
    variant: PropTypes.oneOf(['default', 'elevated', 'outlined', 'ghost', 'gradient']),
    padding: PropTypes.oneOf(['none', 'sm', 'default', 'lg', 'xl']),
    children: PropTypes.node,
};

CardHeader.propTypes = {
    className: PropTypes.string,
    children: PropTypes.node,
};

CardTitle.propTypes = {
    className: PropTypes.string,
    children: PropTypes.node,
};

CardDescription.propTypes = {
    className: PropTypes.string,
    children: PropTypes.node,
};

CardContent.propTypes = {
    className: PropTypes.string,
    children: PropTypes.node,
};

CardFooter.propTypes = {
    className: PropTypes.string,
    children: PropTypes.node,
};


export {
    Card,
    CardHeader,
    CardFooter,
    CardTitle,
    CardDescription,
    CardContent,
    cardVariants,
};