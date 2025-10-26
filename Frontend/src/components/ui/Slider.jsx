// src/components/ui/Slider.jsx

import React from "react";
import { cn } from "../../utils/cn";

/**
 * Lightweight slider inspired by shadcn UI structure but implemented without Radix.
 * Renders a visual track + range fill and a visible thumb while using a native
 * input[type=range] for accessibility and drag behavior.
 */
const Slider = React.forwardRef(
  (
    {
      className,
      value,
      onValueChange,
      min = 0,
      max = 100,
      step = 1,
      thumbSize = "md",
      ...props
    },
    ref
  ) => {
    const internal = Array.isArray(value)
      ? Number(value[0])
      : Number(value) || 0;

    const rangeMin = Number(min) || 0;
    const rangeMax = Number(max) || 100;
    const pct =
      rangeMax > rangeMin
        ? ((internal - rangeMin) / (rangeMax - rangeMin)) * 100
        : 0;
    const filled = Math.max(0, Math.min(100, pct));

    const handleChange = (e) => {
      const v = Number(e.target.value);
      if (onValueChange) onValueChange([v]);
    };

    // map thumb size keyword to px
    const thumbMap = { sm: 6, md: 10, lg: 14 };
    const thumbSizePx = thumbMap[thumbSize] || thumbMap.md;

    return (
      <div
        className={cn(
          "relative flex w-full touch-none select-none items-center group vv-slider",
          className
        )}
      >
        {/* Hide native thumbs across browsers but keep the native input interactive for keyboard/accessibility.
            We render a small scoped CSS block so we don't rely on a global stylesheet change. */}
        <style>{`\
          /* Remove native appearance but keep element interactive */\n
          .vv-slider input[type=range] { -webkit-appearance: none; appearance: none; background: transparent; outline: none; }\n
          /* WebKit/Blink: hide native thumb and track visuals */\n
          .vv-slider input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 0; height: 0; background: transparent; border: none; box-shadow: none; }\n
          .vv-slider input[type=range]::-webkit-slider-runnable-track { background: transparent; border: none; }\n
          /* Firefox: hide thumb */\n
          .vv-slider input[type=range]::-moz-range-thumb { visibility: hidden; width: 0; height: 0; border: none; background: transparent; }\n
          .vv-slider input[type=range]::-moz-range-track { background: transparent; border: none; }\n
          /* IE/Edge legacy */\n
          .vv-slider input[type=range]::-ms-thumb { width: 0; height: 0; visibility: hidden; }\n
          .vv-slider input[type=range]::-ms-track { background: transparent; border: none; }\n
          /* Keep focus outline accessible on the native input but invisible; we rely on the visible thumb scale for focus state */\n
          .vv-slider input[type=range]:focus { outline: none; }\n
        `}</style>
        {/* Visual track */}
        <div
          className="relative h-1.5 w-full grow overflow-hidden rounded-full"
          style={{ background: "var(--slider-empty, rgba(255,255,255,0.08))" }}
        >
          <div
            className="absolute left-0 top-0 h-full"
            style={{ width: `${filled}%`, background: "var(--color-primary)" }}
          />
        </div>

        {/* Visible thumb */}
        <div
          className="pointer-events-none absolute top-1/2 -translate-y-1/2"
          style={{ left: `calc(${filled}% - ${thumbSizePx / 2}px)` }}
          aria-hidden
        >
          <div
            className="rounded-full border shadow transition-transform duration-150 transform group-focus-within:scale-110"
            style={{
              width: `${thumbSizePx}px`,
              height: `${thumbSizePx}px`,
              background: "var(--color-primary)",
              borderColor: "rgba(255,255,255,0.12)",
            }}
          />
        </div>

        {/* Native range for accessibility and interaction (transparent) */}
        <input
          ref={ref}
          type="range"
          min={min}
          max={max}
          step={step}
          value={internal}
          onChange={handleChange}
          className="absolute inset-0 h-full w-full appearance-none bg-transparent cursor-pointer"
          {...props}
        />
      </div>
    );
  }
);

Slider.displayName = "Slider";

export { Slider };
