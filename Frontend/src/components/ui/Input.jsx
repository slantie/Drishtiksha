/**
 * @file src/components/ui/Input.jsx
 * @description A flexible and accessible input component with variants, icons, and password visibility toggle.
 */

"use client";

import React from "react";
import PropTypes from "prop-types"; // ADDED: For runtime type validation
import { cva } from "class-variance-authority";
import { cn } from "../../lib/utils";
import { Eye, EyeOff } from "lucide-react";

// CVA definition remains the same
const inputVariants = cva(
    "flex w-full rounded-lg border bg-white dark:bg-dark-secondary transition-all duration-200 file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-secondary-main dark:placeholder:text-dark-tertiary focus:outline-none disabled:cursor-not-allowed disabled:opacity-50",
    {
        variants: {
            variant: {
                default: "border-light-secondary dark:border-dark-secondary focus:ring-2 focus:ring-primary-main/50 focus:border-primary-main dark:focus:border-primary-main",
                error: "border-red-500 dark:border-red-500 focus:ring-2 focus:ring-red-500/50 focus:border-red-500", // Adjusted to standard colors
                success: "border-green-500 dark:border-green-500 focus:ring-2 focus:ring-green-500/50 focus:border-green-500", // Adjusted to standard colors
            },
            size: {
                default: "h-10 px-3 py-2 text-sm",
                sm: "h-8 px-2 py-1 text-xs",
                lg: "h-12 px-4 py-3 text-base",
                xl: "h-14 px-4 py-3.5 text-lg",
            },
        },
        defaultVariants: {
            variant: "default",
            size: "default",
        },
    }
);

const Input = React.forwardRef(
    (
        {
            className,
            variant,
            size,
            type,
            label,
            error,
            helperText,
            leftIcon,
            rightIcon,
            showPasswordToggle,
            id,
            ...props
        },
        ref
    ) => {
        const [isPasswordVisible, setIsPasswordVisible] = React.useState(false);
        const [inputType, setInputType] = React.useState(type);

        const uniqueId = React.useId();
        const inputId = id || `input-${uniqueId}`;

        React.useEffect(() => {
            if (showPasswordToggle && type === "password") {
                setInputType(isPasswordVisible ? "text" : "password");
            }
        }, [isPasswordVisible, showPasswordToggle, type]);

        const inputVariant = error ? "error" : variant;

        return (
            <div className="space-y-1 w-full">
                {label && (
                    <label
                        htmlFor={inputId}
                        className="text-xs sm:text-sm font-medium text-light-text dark:text-dark-text block"
                    >
                        {label}
                    </label>
                )}
                <div className="relative">
                    {leftIcon && (
                        <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-light-muted-text dark:text-dark-noisy-text">
                            {leftIcon}
                        </div>
                    )}
                    <input
                        type={inputType}
                        className={cn(
                            inputVariants({
                                variant: inputVariant,
                                size,
                                className,
                            }),
                            leftIcon && "pl-10",
                            (rightIcon || showPasswordToggle) && "pr-10"
                        )}
                        ref={ref}
                        id={inputId}
                        {...props}
                    />
                    {(rightIcon || (showPasswordToggle && type === "password")) && (
                        <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                            {showPasswordToggle && type === "password" ? (
                                <button
                                    type="button"
                                    onClick={() => setIsPasswordVisible(!isPasswordVisible)}
                                    className="text-light-muted-text dark:text-dark-noisy-text hover:text-light-text dark:hover:text-dark-text transition-colors focus:outline-none"
                                    tabIndex={-1}
                                >
                                    {isPasswordVisible ? (
                                        <EyeOff className="h-4 w-4" />
                                    ) : (
                                        <Eye className="h-4 w-4" />
                                    )}
                                </button>
                            ) : (
                                rightIcon
                            )}
                        </div>
                    )}
                </div>
                {(error || helperText) && (
                    <p className={cn("text-xs", error ? "text-red-500 dark:text-red-500" : "text-light-muted-text dark:text-dark-noisy-text")}>
                        {error || helperText}
                    </p>
                )}
            </div>
        );
    }
);

Input.displayName = "Input";

// --- PropTypes and DefaultProps ---

Input.propTypes = {
    className: PropTypes.string,
    variant: PropTypes.oneOf(['default', 'error', 'success']),
    size: PropTypes.oneOf(['default', 'sm', 'lg', 'xl']),
    type: PropTypes.string,
    label: PropTypes.string,
    error: PropTypes.string,
    helperText: PropTypes.string,
    leftIcon: PropTypes.node,
    rightIcon: PropTypes.node,
    showPasswordToggle: PropTypes.bool,
    id: PropTypes.string,
};

Input.defaultProps = {
    showPasswordToggle: false,
    type: 'text',
    variant: 'default',
    size: 'default',
};

export { Input, inputVariants };