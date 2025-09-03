// src/components/media/DeleteMediaModal.jsx

import React from "react";
import { Trash2, AlertTriangle } from "lucide-react";
import { Button } from "../ui/Button";
import { Modal } from "../ui/Modal";
import { useDeleteMediaMutation } from "../../hooks/useMediaQuery.jsx";
// REMOVED: import { useNavigate } from "react-router-dom"; // Not directly used here, handled by hook

export const DeleteMediaModal = ({ isOpen, onClose, media }) => {
  const deleteMutation = useDeleteMediaMutation();
  // REMOVED: const navigate = useNavigate(); // Navigation is handled in useDeleteMediaMutation hook

  const handleDelete = async () => {
    if (!media?.id) {
      console.error("DeleteMediaModal: Media ID is missing.");
      onClose();
      return;
    }
    // The useDeleteMediaMutation hook already handles the API call, toast,
    // cache invalidation, and navigation back to dashboard on success.
    await deleteMutation.mutateAsync(media.id);
    onClose(); // Close the modal regardless of success/failure (toast handles result)
  };

  // Render nothing if not open or no media is provided
  if (!isOpen || !media) return null;

  const modalFooter = (
    <>
      <Button
        variant="outline"
        onClick={onClose}
        disabled={deleteMutation.isPending}
      >
        Cancel
      </Button>
      <Button
        variant="destructive"
        onClick={handleDelete}
        isLoading={deleteMutation.isPending}
      >
        {!deleteMutation.isPending && <Trash2 className="mr-2 h-4 w-4" />}
        {deleteMutation.isPending ? "Deleting..." : "Delete Media"}
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
          <AlertTriangle className="h-6 w-6 text-red-600" aria-hidden="true" />
        </div>
        <div className="text-sm">
          <p className="font-semibold">
            Are you sure you want to delete{" "}
            <span className="font-bold text-light-text dark:text-dark-text">
              {media.filename}
            </span>
            ?
          </p>
          <p className="mt-1 text-light-muted-text dark:text-dark-muted-text">
            This will permanently remove the file and all of its associated
            analysis data. This action cannot be undone.
          </p>
        </div>
      </div>
    </Modal>
  );
};
