// src/components/media/DeleteMediaModal.jsx

import React from "react";
import { Trash2, AlertTriangle } from "lucide-react";
import { Button } from "../ui/Button";
import { Modal } from "../ui/Modal";
import { useDeleteMediaMutation } from "../../hooks/useMediaQuery.jsx";
import { useNavigate } from "react-router-dom";

export const DeleteMediaModal = ({ isOpen, onClose, media }) => {
    const deleteMutation = useDeleteMediaMutation();
    const navigate = useNavigate();

    const handleDelete = async () => {
        if (!media) return;
        // The hook now handles navigation on success.
        await deleteMutation.mutateAsync(media.id);
        onClose();
    };

    if (!isOpen || !media) return null;

    const modalFooter = (
        <>
            <Button variant="outline" onClick={onClose} disabled={deleteMutation.isPending}>
                Cancel
            </Button>
            <Button variant="destructive" onClick={handleDelete} isLoading={deleteMutation.isPending}>
                {!deleteMutation.isPending && <Trash2 className="mr-2 h-4 w-4" />}
                Delete Media
            </Button>
        </>
    );

    return (
        <Modal isOpen={isOpen} onClose={onClose} title="Confirm Deletion" footer={modalFooter}>
            <div className="flex items-start space-x-4">
                <div className="flex-shrink-0 flex items-center justify-center h-12 w-12 rounded-full bg-red-100 dark:bg-red-900/30 sm:mx-0 sm:h-10 sm:w-10">
                    <AlertTriangle className="h-6 w-6 text-red-600" aria-hidden="true" />
                </div>
                <div className="text-sm">
                    <p>
                        Are you sure you want to delete{" "}
                        <span className="font-semibold">{media.filename}</span>?
                        This will permanently remove the file and all of its analysis data. This action cannot be undone.
                    </p>
                </div>
            </div>
        </Modal>
    );
};