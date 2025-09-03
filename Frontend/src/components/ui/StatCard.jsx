// src/components/ui/StatCard.jsx
"use client";

import React from "react";
import PropTypes from "prop-types";
import { Card } from "./Card"; // Use the shared Card component
import CountUp from "react-countup";
import { AlertCircle } from "lucide-react";
import clsx from "clsx";
import { LoadingSpinner } from "./LoadingSpinner.jsx"; // Import the unified LoadingSpinner

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
      border: "hover:border-primary-main",
      background: "bg-primary-main",
      text: "text-primary-main",
      bgLight: "bg-primary-main/10", // For internal elements
    },
    blue: {
      border: "hover:border-blue-500",
      background: "bg-blue-500",
      text: "text-blue-500",
      bgLight: "bg-blue-500/10",
    },
    purple: {
      border: "hover:border-purple-500",
      background: "bg-purple-500",
      text: "text-purple-500",
      bgLight: "bg-purple-500/10",
    },
    green: {
      border: "hover:border-green-500",
      background: "bg-green-500",
      text: "text-green-500",
      bgLight: "bg-green-500/10",
    },
    red: {
      border: "hover:border-red-600",
      background: "bg-red-600",
      text: "text-red-600",
      bgLight: "bg-red-600/10",
    },
    yellow: {
      // Added yellow for potential warning/processing status
      border: "hover:border-yellow-500",
      background: "bg-yellow-500",
      text: "text-yellow-500",
      bgLight: "bg-yellow-500/10",
    },
  };

  // Select the theme based on cardColor prop, defaulting to 'primary'
  const theme = colorStyles[cardColor] || colorStyles.primary;

  return (
    <Card // Using the imported Card component
      onClick={onClick}
      className={clsx(
        "p-4 relative overflow-hidden transition-all duration-300 rounded-xl", // Changed to rounded-xl for consistency with Card.jsx
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
                <AlertCircle className={clsx("w-8 h-8 mr-2", theme.text)} />
                <span className={clsx("text-lg", theme.text)}>Error</span>
              </div>
            ) : isLoading ? (
              <LoadingSpinner
                size="md"
                spinnerClassName={theme.text}
                className="items-start"
              /> // Use unified LoadingSpinner
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
            <p className={clsx("text-xs mt-1", theme.text, "opacity-70")}>
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
  cardColor: PropTypes.oneOf([
    "primary",
    "blue",
    "purple",
    "green",
    "red",
    "yellow",
  ]), // Added yellow
};

StatCard.defaultProps = {
  onClick: undefined,
  isLoading: false,
  error: null,
  subtitle: "",
  className: "",
  cardColor: "primary", // Default theme is 'primary'
};
