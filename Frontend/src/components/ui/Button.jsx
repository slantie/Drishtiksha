// src/components/ui/Button.jsx

import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cn } from "../../lib/utils";
import { Loader2 } from "lucide-react";
import { buttonVariants } from "../../constants/buttonConstants";

/**
 * A versatile and themeable button component with loading and primary/secondary variants.
 * @param {object} props - The component props.
 * @param {string} [props.variant] - The visual style of the button.
 * @param {string} [props.size] - The size of the button.
 * @param {boolean} [props.asChild] - Renders the component as a child of the element it's passed to.
 * @param {boolean} [props.isLoading] - If true, shows a loading spinner and disables the button.
 * @param {string} [props.className] - Additional CSS classes to apply.
 */
const Button = React.forwardRef(
  (
    {
      className,
      variant,
      size,
      asChild = false,
      isLoading = false,
      children,
      ...props
    },
    ref
  ) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        disabled={isLoading || props.disabled}
        {...props}
      >
        {isLoading ? (
          // Render a single span containing both loader and (optionally hidden) text
          <>
            <Loader2
              className={cn("h-5 w-5 animate-spin", children && "mr-2")}
            />
            {children && (
              <span className={cn(isLoading && "sr-only")}>{children}</span>
            )}
            {!children && <span className="sr-only">Loading...</span>}
          </>
        ) : (
          // Render children directly when not loading
          children
        )}
      </Comp>
    );
  }
);
Button.displayName = "Button";

export { Button };
