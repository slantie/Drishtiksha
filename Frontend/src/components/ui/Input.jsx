// src/components/ui/Input.jsx

import * as React from "react";
import { cn } from "../../lib/utils";

const Input = React.forwardRef(
  ({ className, type, label, leftIcon, rightIcon, id, ...props }, ref) => {
    // Added 'label' and 'id'
    const inputId = id || React.useId(); // Generate unique ID if not provided

    return (
      <div className="relative w-full">
        {label && (
          <label
            htmlFor={inputId}
            className="block text-sm font-medium text-light-muted-text dark:text-dark-muted-text mb-1"
          >
            {label}
          </label>
        )}
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
          id={inputId} // Connect label to input
          type={type}
          className={cn(
            "flex h-10 w-full rounded-lg border border-light-secondary dark:border-dark-secondary bg-transparent py-2 text-sm",
            "file:border-0 file:bg-transparent file:text-sm file:font-medium",
            "placeholder:text-light-muted-text dark:placeholder:text-dark-muted-text",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-primary-main",
            "disabled:cursor-not-allowed disabled:opacity-50",
            // REFACTOR: Correctly applied padding based on icon presence.
            leftIcon ? "pl-10" : "px-3",
            rightIcon ? "pr-10" : "px-3", // Ensure enough space for right icon
            className
          )}
          ref={ref}
          {...props}
        />
        {/* REFACTOR: Correctly implemented the rightIcon slot. It is now interactive. */}
        {rightIcon && ( // Check for truthiness instead of <></>
          <div className="absolute inset-y-0 right-0 flex items-center pr-3">
            {/* We assume rightIcon itself is a component or element (e.g., a button with an icon)
                            and it manages its own styling and onClick. We just ensure it's displayed correctly. */}
            {rightIcon}
          </div>
        )}
      </div>
    );
  }
);
Input.displayName = "Input";

export { Input };
