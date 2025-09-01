// src/components/media/EditMediaModal.jsx

import React, { useState, useEffect } from "react";
import { Save } from "lucide-react";
import { Button } from "../ui/Button";
import { Input } from "../ui/Input";
import { Modal } from "../ui/Modal";
import { useUpdateMediaMutation } from '../../hooks/useMediaQuery.jsx';

export const EditMediaModal = ({ isOpen, onClose, media }) => {
    const [description, setDescription] = useState("");
    const updateMutation = useUpdateMediaMutation();

    useEffect(() => {
        if (media) {
            setDescription(media.description || "");
        }
    }, [media]);

    const handleSave = async () => {
        if (!media) return;
        // REFACTOR: We are only updating the description, as the filename is immutable.
        await updateMutation.mutateAsync({ mediaId: media.id, updateData: { description } });
        onClose();
    };

    if (!isOpen || !media) return null;

    const modalFooter = (
        <>
            <Button variant="outline" onClick={onClose} disabled={updateMutation.isPending}>
                Cancel
            </Button>
            <Button onClick={handleSave} isLoading={updateMutation.isPending}>
                {!updateMutation.isPending && <Save className="mr-2 h-4 w-4" />}
                Save Changes
            </Button>
        </>
    );

    return (
        <Modal
            isOpen={isOpen}
            onClose={onClose}
            title="Edit Media Description"
            footer={modalFooter}
        >
            <div className="space-y-4">
                {/* REFACTOR: Removed the filename input field. */}
                <Input
                    as="textarea"
                    label="Description"
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    placeholder="Enter a description for this media file"
                    rows={4}
                    disabled={updateMutation.isPending}
                />
            </div>
        </Modal>
    );
};