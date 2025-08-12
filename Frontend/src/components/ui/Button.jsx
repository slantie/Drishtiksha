/**
 * @file src/components/ui/Button.jsx
 * @description A reusable button component with multiple variants and sizes, styled with Tailwind CSS.
 * This component is based on the shadcn/ui button component.
 */

import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva } from "class-variance-authority";
import { cn } from "../../lib/utils";

/**
 * Defines the button's visual variants and sizes using class-variance-authority.
 */
const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-light-highlight dark:bg-dark-highlight text-white shadow hover:bg-light-highlight/90 dark:hover:bg-dark-highlight/90",
        destructive: "bg-red-500/10 text-red-500 shadow-sm hover:bg-red-500/20 dark:bg-red-500/20 dark:hover:bg-red-500/30",
        outline: "border border-light-secondary dark:border-dark-secondary bg-transparent shadow-sm hover:bg-light-hover dark:hover:bg-dark-hover text-light-text dark:text-dark-text",
        secondary: "bg-light-muted-background dark:bg-dark-secondary text-light-text dark:text-dark-text shadow-sm hover:bg-light-hover dark:hover:bg-dark-hover",
        ghost: "hover:bg-light-hover dark:hover:bg-dark-hover text-light-muted-text dark:text-dark-muted-text hover:text-light-text dark:hover:text-dark-text",
        link: "text-light-highlight dark:text-dark-highlight underline-offset-4 hover:underline",
      },
      size: {
        default: "py-3 px-4",
        sm: "h-8 rounded-md px-3 text-xs",
        lg: "h-10 rounded-md px-8",
        icon: "h-9 w-9",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

/**
 * Button component that can be used as a child component or with different HTML elements.
 * @param {object} props - The component props.
 * @param {string} [props.variant] - The visual style of the button.
 * @param {string} [props.size] - The size of the button.
 * @param {boolean} [props.asChild] - Renders the component as a child of the element it's passed to.
 * @param {string} [props.className] - Additional CSS classes to apply.
 */
const Button = React.forwardRef(({ className, variant, size, asChild = false, ...props }, ref) => {
  const Comp = asChild ? Slot : "button";
  return (
    <Comp
      className={cn(buttonVariants({ variant, size, className }))}
      ref={ref}
      {...props}
    />
  );
});
Button.displayName = "Button";

export { Button, buttonVariants };
