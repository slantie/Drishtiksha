// src/components/ui/Slider.jsx

import React from "react";
import { cn } from "../../utils/cn";

const Slider = React.forwardRef(
  ({ className, value, onValueChange, ...props }, ref) => {
    const internalValue = Array.isArray(value) ? value[0] : 0;

    const handleValueChange = (e) => {
      if (onValueChange) {
        onValueChange([parseFloat(e.target.value)]);
      }
    };

    return (
      <input
        type="range"
        ref={ref}
        value={internalValue}
        onChange={handleValueChange}
        className={cn(
          "w-full h-2 bg-white/20 rounded-full appearance-none cursor-pointer",
          "accent-primary-main",
          "focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-main/50",
          className
        )}
        {...props}
      />
    );
  }
);

Slider.displayName = "Slider";

export { Slider };
