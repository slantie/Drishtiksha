// src/components/ui/Input.jsx

import * as React from "react";
import { cn } from "../../lib/utils";

const Input = React.forwardRef(
  ({ className, type, label, leftIcon, rightIcon, id, error, ...props }, ref) => {
    const generatedId = React.useId();
    const inputId = id || generatedId;
    const errorId = error ? `${inputId}-error` : undefined;

    return (
      <div>
      {label && (
          <label
            htmlFor={inputId}
            className="block text-sm font-medium text-light-muted-text dark:text-dark-muted-text mb-2"
          >
            {label}
          </label>
        )}
      <div className="relative w-full">
        {/* REFACTOR: Left icon is now a purely decorative element with pointer-events-none. */}
        {leftIcon && (
          <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3">
            {React.cloneElement(leftIcon, {
              className:
                "h-5 w-5 text-light-muted-text dark:text-dark-muted-text",
            })}
          </div>
        )}
        <input
          id={inputId}
          type={type}
          className={cn(
            "flex h-10 w-full rounded-lg border bg-transparent py-2 text-sm",
            "file:border-0 file:bg-transparent file:text-sm file:font-medium",
            "placeholder:text-light-muted-text dark:placeholder:text-dark-muted-text",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-primary-main",
            "disabled:cursor-not-allowed disabled:opacity-50",
            error
              ? "border-red-500 dark:border-red-500 focus-visible:ring-red-500"
              : "border-light-secondary dark:border-dark-secondary",
            leftIcon ? "pl-10" : "px-3",
            rightIcon ? "pr-10" : "px-3",
            className
          )}
          ref={ref}
          aria-invalid={error ? "true" : "false"}
          aria-describedby={errorId}
          {...props}
        />
        {/* REFACTOR: Correctly implemented the rightIcon slot. It is now interactive. */}
        {rightIcon && (
          <div className="absolute inset-y-0 right-0 flex items-center pr-3">
            {rightIcon}
          </div>
        )}
      </div>
      {error && (
        <p
          id={errorId}
          className="mt-1.5 text-xs text-red-600 dark:text-red-400"
          role="alert"
        >
          {error}
        </p>
      )}
      </div>
    );
  }
);
Input.displayName = "Input";

export { Input };
