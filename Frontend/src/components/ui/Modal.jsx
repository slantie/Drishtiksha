// src/components/ui/Modal.jsx

import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X } from "lucide-react";
import { Button } from "./Button";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
} from "./Card";
import { cn } from "../../lib/utils"; // Import cn for class merging

/**
 * A reusable modal component for consistent popups.
 * @param {object} props - The component props.
 * @param {boolean} props.isOpen - Controls the visibility of the modal.
 * @param {function} props.onClose - Function to call when the modal is requested to close.
 * @param {string} [props.title] - The title displayed in the modal header.
 * @param {string} [props.description] - A descriptive text displayed below the title.
 * @param {React.ReactNode} props.children - The main content of the modal.
 * @param {React.ReactNode} [props.footer] - Optional content for the modal footer (e.g., action buttons).
 * @param {string} [props.size='lg'] - Controls the maximum width of the modal ('sm', 'md', 'lg', 'xl', '2xl', '3xl', '4xl', '5xl').
 * @param {string} [props.className] - Additional classes for the modal content card.
 */
const Modal = ({
  isOpen,
  onClose,
  title,
  description,
  children,
  footer,
  size = "lg",
  className,
}) => {
  const modalWidthClass =
    {
      sm: "max-w-sm",
      md: "max-w-md",
      lg: "max-w-lg",
      xl: "max-w-xl",
      "2xl": "max-w-2xl",
      "3xl": "max-w-3xl",
      "4xl": "max-w-4xl",
      "5xl": "max-w-5xl",
    }[size] || "max-w-lg"; // Default to max-w-lg

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-[-25px] z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm p-4" // Increased z-index, added p-4 for smaller screens
          onClick={onClose}
          aria-modal="true" // Accessibility
          role="dialog" // Accessibility
        >
          <motion.div
            initial={{ scale: 0.95, y: 20 }}
            animate={{ scale: 1, y: 0 }}
            exit={{ scale: 0.95, y: 20, opacity: 0 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className={cn("relative w-full", modalWidthClass)} // Apply dynamic max-width
            onClick={(e) => e.stopPropagation()}
          >
            <Card className={cn("overflow-hidden", className)}>
              {" "}
              {/* Added overflow-hidden for rounded corners */}
              <CardHeader>
                <div className="flex justify-between items-start">
                  <div>
                    {title && <CardTitle>{title}</CardTitle>}
                    {description && (
                      <CardDescription className="mt-1">
                        {description}
                      </CardDescription>
                    )}
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={onClose}
                    className="-mt-2 -mr-2"
                    aria-label="Close modal" // Accessibility
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              </CardHeader>
              <CardContent>{children}</CardContent>
              {footer && (
                <CardFooter className="justify-end space-x-2">
                  {footer}
                </CardFooter>
              )}
            </Card>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export { Modal };
