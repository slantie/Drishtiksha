// src/components/videos/DeleteVideoModal.jsx

import React, { useState } from "react";
import { Trash2, X, Loader2, AlertTriangle } from "lucide-react";
import { Button } from "../ui/Button";

export const DeleteVideoModal = ({ isOpen, onClose, video, onDelete }) => {
    const [isDeleting, setIsDeleting] = useState(false);

    const handleDelete = async () => {
        setIsDeleting(true);
        try {
            await onDelete(video.id);
            onClose();
        } catch (error) {
            console.log("Delete failed:", error);
        } finally {
            setIsDeleting(false);
        }
    };

    if (!isOpen || !video) return null;

    return (
        <div className="fixed inset-[-25px] z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
            <div className="relative w-full max-w-md bg-light-background dark:bg-dark-background rounded-3xl p-8 shadow-2xl border border-red-500/20">
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 p-2 rounded-full hover:bg-light-muted-background dark:hover:bg-dark-muted-background"
                >
                    <X />
                </button>
                <div className="text-center">
                    <AlertTriangle className="w-12 h-12 text-red-500 mx-auto mb-4" />
                    <h2 className="text-2xl font-bold mb-2">
                        Confirm Deletion
                    </h2>
                    <p>
                        Are you sure you want to delete "{video.filename}"? This
                        action cannot be undone.
                    </p>
                    <div className="flex justify-center space-x-4 mt-6">
                        <Button
                            variant="outline"
                            onClick={onClose}
                            disabled={isDeleting}
                        >
                            Cancel
                        </Button>
                        <Button
                            onClick={handleDelete}
                            disabled={isDeleting}
                            className="bg-red-500 hover:bg-red-600 text-white"
                        >
                            {isDeleting ? (
                                <>
                                    <Loader2 className="animate-spin mr-2" />{" "}
                                    Deleting...
                                </>
                            ) : (
                                <>
                                    <Trash2 className="mr-2" /> Delete
                                </>
                            )}
                        </Button>
                    </div>
                </div>
            </div>
        </div>
    );
};
