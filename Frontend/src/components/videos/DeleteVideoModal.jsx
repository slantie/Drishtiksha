// src/components/videos/DeleteVideoModal.jsx

import React, { useState } from "react";
import { Trash2, AlertTriangle } from "lucide-react";
import { Button } from "../ui/Button";
import { Modal } from "../ui/Modal"; // REFACTOR: Using the new base Modal component.

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

    // REFACTOR: Using the destructive variant for the primary button.
    const modalFooter = (
        <>
            <Button variant="outline" onClick={onClose} disabled={isDeleting}>
                Cancel
            </Button>
            <Button
                variant="destructive"
                onClick={handleDelete}
                isLoading={isDeleting}
            >
                {!isDeleting && <Trash2 className="mr-2 h-4 w-4" />}
                Delete Video
            </Button>
        </>
    );

    return (
        <Modal
            isOpen={isOpen}
            onClose={onClose}
            title="Confirm Deletion"
            footer={modalFooter}
        >
            <div className="flex items-start space-x-4">
                <div className="flex-shrink-0 flex items-center justify-center h-12 w-12 rounded-full bg-red-100 dark:bg-red-900/30 sm:mx-0 sm:h-10 sm:w-10">
                    <AlertTriangle
                        className="h-6 w-6 text-red-600"
                        aria-hidden="true"
                    />
                </div>
                <div className="text-sm">
                    <p>
                        Are you sure you want to delete{" "}
                        <span className="font-semibold">{video.filename}</span>?
                        This action cannot be undone.
                    </p>
                </div>
            </div>
        </Modal>
    );
};
