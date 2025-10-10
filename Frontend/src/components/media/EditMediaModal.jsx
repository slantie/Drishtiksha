// src/components/media/EditMediaModal.jsx

import React, { useState, useEffect } from "react";
import { Save, FilePen } from "lucide-react"; // Added FilePen icon
import { Button } from "../ui/Button";
import { Input } from "../ui/Input";
import { Modal } from "../ui/Modal";
import { useUpdateMediaMutation } from "../../hooks/useMediaQuery.jsx";

export const EditMediaModal = ({ isOpen, onClose, media }) => {
  const [description, setDescription] = useState("");
  const updateMutation = useUpdateMediaMutation();

  useEffect(() => {
    if (media) {
      setDescription(media.description || "");
    } else {
      // Reset description if media prop becomes null/undefined
      setDescription("");
    }
  }, [media]);

  const handleSave = async () => {
    if (!media?.id) {
      console.error("EditMediaModal: Media ID is missing.");
      onClose();
      return;
    }
    // REFACTOR: We are only updating the description, as the filename is immutable.
    await updateMutation.mutateAsync({
      mediaId: media.id,
      updateData: { description },
    });
    onClose(); // Close modal after mutation (success/failure handled by toast)
  };

  // Render nothing if not open or no media is provided
  if (!isOpen || !media) return null;

  const modalFooter = (
    <>
      <Button
        variant="outline"
        onClick={onClose}
        disabled={updateMutation.isPending}
      >
        Cancel
      </Button>
      <Button onClick={handleSave} isLoading={updateMutation.isPending}>
        {!updateMutation.isPending && <Save className="mr-2 h-4 w-4" />}
        {updateMutation.isPending ? "Saving..." : "Save Changes"}
      </Button>
    </>
  );

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={`Edit Description for "${media.filename || "media file"}"`} // Dynamic title
      footer={modalFooter}
    >
      <div className="space-y-4">
        <Input
          as="textarea"
          label="Description"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Enter a description for this media file (max 500 characters)"
          rows={4}
          maxLength={500} // Add max length client-side hint
          disabled={updateMutation.isPending}
          leftIcon={<FilePen className="h-5 w-5" />} // Added icon
          rightIcon={<></>}
        />
      </div>
    </Modal>
  );
};
