// src/components/dashboard/StatCard.jsx
"use client";

import React from "react";
import PropTypes from "prop-types";
import { Card } from "./Card";
import CountUp from "react-countup";
import { AlertCircle } from "lucide-react";
import clsx from "clsx";

// A simple Card component for context, as it was imported.
// In a real app, this would be in its own file.
const BaseCard = React.forwardRef(({ className, ...props }, ref) => (
    <div
        ref={ref}
        className={clsx(
            "rounded-lg border bg-card text-card-foreground shadow-sm",
            className
        )}
        {...props}
    />
));
BaseCard.displayName = "Card";

export const StatCard = ({
    title,
    value,
    icon: Icon,
    onClick,
    isLoading,
    error,
    subtitle,
    className,
    cardColor,
}) => {
    const isClickable = !!onClick;

    // --- Dynamic Theming ---
    // This map defines the Tailwind classes for each color theme.
    // This approach is friendly to Tailwind's JIT compiler, as it sees full class names.
    const colorStyles = {
        primary: {
            border: "hover:border-primary-main dark:hover:border-primary-light",
            background: "bg-primary-main dark:bg-primary-light",
            text: "text-primary-main dark:text-primary-light",
        },
        blue: {
            border: "hover:border-blue-500 dark:hover:border-blue-400",
            background: "bg-blue-500 dark:bg-blue-400",
            text: "text-blue-500 dark:text-blue-400",
        },
        purple: {
            border: "hover:border-purple-500 dark:hover:border-purple-400",
            background: "bg-purple-500 dark:bg-purple-400",
            text: "text-purple-500 dark:text-purple-400",
        },
        green: {
            border: "hover:border-green-500 dark:hover:border-green-400",
            background: "bg-green-500 dark:bg-green-400",
            text: "text-green-500 dark:text-green-400",
        },
        red: {
            border: "hover:border-red-600 dark:hover:border-red-500",
            background: "bg-red-600 dark:bg-red-500",
            text: "text-red-600 dark:text-red-500",
        },
    };

    // Select the theme based on cardColor prop, defaulting to 'primary'
    const theme = colorStyles[cardColor] || colorStyles.primary;

    return (
        <Card // Using the placeholder BaseCard component
            onClick={onClick}
            className={clsx(
                "p-4 relative overflow-hidden transition-all duration-300 rounded-2xl",
                "bg-light-background dark:bg-dark-muted-background",
                "border border-light-secondary dark:border-dark-secondary",
                "flex flex-col justify-between",
                {
                    "cursor-pointer group hover:shadow-lg": isClickable,
                    [theme.border]: isClickable,
                },
                className
            )}
            {...(isClickable && {
                role: "button",
                tabIndex: 0,
                "aria-label": `View details for ${title}`,
            })}
        >
            {/* Animated left border with dynamic background color */}
            {isClickable && (
                <div
                    className={clsx(
                        "absolute top-0 left-0 w-2 h-full transform -translate-x-full group-hover:translate-x-0 transition-transform duration-300",
                        theme.background
                    )}
                ></div>
            )}

            <div className="py-2 flex ml-2 justify-between items-start gap-4">
                <div className="flex-grow">
                    <p className="text-sm font-semibold text-light-muted-text dark:text-dark-muted-text">
                        {title}
                    </p>
                    <div
                        className={clsx(
                            "text-4xl md:text-5xl font-extrabold transition-colors duration-300",
                            theme.text, // Always use theme color for the value
                            {
                                "group-hover:opacity-80": isClickable,
                            }
                        )}
                    >
                        {error ? (
                            <div className="flex items-center" title={error}>
                                <AlertCircle
                                    className={clsx("w-8 h-8 mr-2", theme.text)}
                                />
                                <span className={clsx("text-lg", theme.text)}>
                                    Error
                                </span>
                            </div>
                        ) : isLoading ? (
                            <div className="flex items-center">
                                <div
                                    className={clsx(
                                        "animate-spin rounded-full h-10 w-10 border-4 border-current border-t-transparent",
                                        theme.text
                                    )}
                                ></div>
                            </div>
                        ) : typeof value === "number" ? (
                            <CountUp
                                end={value}
                                duration={2}
                                separator=","
                                enableScrollSpy={true}
                                scrollSpyOnce={true}
                                className={clsx(
                                    "md:text-4xl lg:text-6xl font-black",
                                    theme.text
                                )}
                            />
                        ) : (
                            <span className={theme.text}>{value}</span>
                        )}
                    </div>
                    {subtitle && (
                        <p
                            className={clsx(
                                "text-xs mt-1",
                                theme.text,
                                "opacity-70"
                            )}
                        >
                            {subtitle}
                        </p>
                    )}
                </div>
                {/* Icon with dynamic color and opacity for hover effect */}
                <Icon
                    className={clsx(
                        "h-20 w-20 transition-all duration-300 group-hover:scale-110 flex-shrink-0",
                        theme.text, // Apply dynamic text color
                        "opacity-80 group-hover:opacity-100" // Control opacity on hover
                    )}
                />
            </div>
        </Card>
    );
};

StatCard.propTypes = {
    title: PropTypes.string.isRequired,
    value: PropTypes.oneOfType([PropTypes.number, PropTypes.string]).isRequired,
    icon: PropTypes.elementType.isRequired,
    onClick: PropTypes.func,
    isLoading: PropTypes.bool,
    error: PropTypes.string,
    subtitle: PropTypes.string,
    className: PropTypes.string,
    /** The color theme of the card */
    cardColor: PropTypes.oneOf(["primary", "blue", "purple", "green", "red"]),
};

StatCard.defaultProps = {
    onClick: undefined,
    isLoading: false,
    error: null,
    subtitle: "",
    className: "",
    cardColor: "primary", // Default theme is 'primary'
};
