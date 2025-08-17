// src/components/ui/Modal.jsx

// NEW COMPONENT: A reusable modal wrapper for consistent popups.
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

const Modal = ({ isOpen, onClose, title, description, children, footer }) => {
    return (
        <AnimatePresence>
            {isOpen && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="fixed inset-[-25px] z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
                    onClick={onClose}
                >
                    <motion.div
                        initial={{ scale: 0.95, y: 20 }}
                        animate={{ scale: 1, y: 0 }}
                        exit={{ scale: 0.95, y: 20, opacity: 0 }}
                        transition={{ duration: 0.2, ease: "easeOut" }}
                        className="relative w-full max-w-lg"
                        onClick={(e) => e.stopPropagation()}
                    >
                        <Card>
                            <CardHeader>
                                <div className="flex justify-between items-start">
                                    <div>
                                        {title && (
                                            <CardTitle>{title}</CardTitle>
                                        )}
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
                                    >
                                        <X className="h-4 w-4" />
                                        <span className="sr-only">Close</span>
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
