// src/components/ui/Button.jsx

import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva } from "class-variance-authority";
import { cn } from "../../lib/utils";
import { Loader2 } from "lucide-react";

// REFACTOR: Variants are now explicitly designed for a Primary/Secondary UI pattern.
// Consistent focus rings and transitions have been applied to all variants for a polished feel.
const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap text-sm font-semibold transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-primary-main disabled:pointer-events-none disabled:opacity-60",
  {
    variants: {
      variant: {
        // Primary Action Button: Use for the main call-to-action on a page.
        default:
          "bg-primary-main text-white shadow-sm hover:bg-primary-main/90 rounded-full",
        // Destructive Action Button: Use for actions that delete data or are irreversible.
        destructive:
          "bg-red-600/25 text-red-500 shadow-sm hover:bg-red-600/40 rounded-full",
        // Secondary Action Button: Use for the less important action in a pair.
        outline:
          "border border-light-secondary dark:border-dark-secondary bg-transparent hover:bg-light-hover dark:hover:bg-dark-hover rounded-full",
        // Alternative Secondary Button: A subtle alternative to the outline style.
        secondary:
          "bg-light-muted-background dark:bg-dark-secondary text-light-text dark:text-dark-text hover:bg-light-hover dark:hover:bg-dark-hover rounded-full",
        // Ghost Button: Use for low-emphasis actions, often within other components like tables.
        ghost: "hover:bg-light-hover dark:hover:bg-dark-hover rounded-full",
        // Link Button: Use for actions that should look like a hyperlink.
        link: "text-primary-main underline-offset-4 hover:underline rounded-full",
      },
      size: {
        default: "h-10 px-4 py-2 rounded-full",
        sm: "h-9 px-3 rounded-full ",
        lg: "h-11 px-8 rounded-full",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

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
            <Loader2 className={cn("h-5 w-5 animate-spin", children && "mr-2")} />
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

export { Button, buttonVariants };