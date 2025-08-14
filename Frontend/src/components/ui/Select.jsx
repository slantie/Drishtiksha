// src/components/ui/Select.jsx

import React, { forwardRef, useState } from "react";
import { ChevronDown } from "lucide-react";
import PropTypes from "prop-types";

const Select = forwardRef(
    (
        { className = "", label, id, name, children, onChange, ...props },
        ref
    ) => {
        const [isOpen, setIsOpen] = useState(false);

        const handleFocus = () => {
            setIsOpen(true);
        };

        const handleBlur = () => {
            setIsOpen(false);
        };

        const handleMouseDown = () => {
            setIsOpen(true);
        };

        const handleChange = (e) => {
            // Reset to closed state after selection
            setTimeout(() => setIsOpen(false), 100);
            // Call the original onChange if provided
            if (onChange) {
                onChange(e);
            }
        };

        return (
            <div className="space-y-1">
                {label && (
                    <label
                        htmlFor={id}
                        className="block text-sm font-medium text-light-text dark:text-dark-text"
                    >
                        {label}
                    </label>
                )}
                <div className="relative">
                    <select
                        ref={ref}
                        id={id}
                        name={name}
                        onFocus={handleFocus}
                        onBlur={handleBlur}
                        onMouseDown={handleMouseDown}
                        onChange={handleChange}
                        className={`
                        block w-full px-3 py-2 pr-10
                        bg-light-background dark:bg-dark-background 
                        text-light-text dark:text-dark-text 
                        border border-light-secondary dark:border-dark-secondary 
                        rounded-lg shadow-sm
                        focus:outline-none focus:ring-2 focus:ring-primary-main focus:border-primary-main
                        disabled:opacity-50 disabled:cursor-not-allowed
                        text-sm transition-colors appearance-none
                        ${className}
                    `}
                        {...props}
                    >
                        {children}
                    </select>
                    {/* Custom arrow icon positioned absolutely */}
                    <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-2">
                        <ChevronDown
                            className={`h-5 w-5 text-light-muted-text dark:text-dark-muted-text transition-transform duration-200 ease-in-out ${
                                isOpen ? "rotate-180" : "rotate-0"
                            }`}
                            aria-hidden="true"
                        />
                    </div>
                </div>
            </div>
        );
    }
);

Select.displayName = "Select";

Select.propTypes = {
    className: PropTypes.string,
    label: PropTypes.string,
    id: PropTypes.string,
    name: PropTypes.string,
    children: PropTypes.node.isRequired,
    onChange: PropTypes.func,
};

export { Select };
