// src/components/ui/Input.jsx

import * as React from "react";
import { cn } from "../../lib/utils";

const Input = React.forwardRef(
    ({ className, type, leftIcon, rightIcon, ...props }, ref) => {
        return (
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
                    type={type}
                    className={cn(
                        "flex h-10 w-full rounded-lg border border-light-secondary dark:border-dark-secondary bg-transparent py-2 text-sm",
                        "file:border-0 file:bg-transparent file:text-sm file:font-medium",
                        "placeholder:text-light-muted-text dark:placeholder:text-dark-muted-text",
                        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-primary-main",
                        "disabled:cursor-not-allowed disabled:opacity-50",
                        // REFACTOR: Correctly applied padding based on icon presence.
                        leftIcon ? "pl-10" : "px-3",
                        rightIcon ? "pr-10" : "px-3",
                        className
                    )}
                    ref={ref}
                    {...props}
                />
                {/* REFACTOR: Correctly implemented the rightIcon slot. It is now interactive. */}
                {rightIcon && rightIcon !== <></> && (
                    <div className="absolute inset-y-0 right-0 flex items-center pr-3">
                        {React.cloneElement(rightIcon, {
                            className:
                                "h-6 w-6 text-light-muted-text dark:text-dark-muted-text hover:text-primary-main transition-colors",
                        })}
                    </div>
                )}
            </div>
        );
    }
);
Input.displayName = "Input";

export { Input };
