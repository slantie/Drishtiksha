// src/components/media/EditMediaModal.jsx

import React, { useState, useEffect } from "react";
import { Save } from "lucide-react";
import { Button } from "../ui/Button";
import { Input } from "../ui/Input";
import { Modal } from "../ui/Modal";

// RENAMED: from EditVideoModal to EditMediaModal
export const EditMediaModal = ({ isOpen, onClose, media, onUpdate }) => {
    const [filename, setFilename] = useState("");
    const [description, setDescription] = useState("");
    const [isSaving, setIsSaving] = useState(false);

    useEffect(() => {
        if (media) {
            setFilename(media.filename || "");
            setDescription(media.description || "");
        }
    }, [media]);

    const handleSave = async () => {
        setIsSaving(true);
        try {
            await onUpdate(media.id, { filename, description });
            onClose();
        } catch (error) {
            console.error("Update failed:", error);
        } finally {
            setIsSaving(false);
        }
    };

    if (!isOpen || !media) return null;

    const modalFooter = (
        <>
            <Button variant="outline" onClick={onClose}>
                Cancel
            </Button>
            <Button onClick={handleSave} isLoading={isSaving}>
                {!isSaving && <Save className="mr-2 h-4 w-4" />}
                Save Changes
            </Button>
        </>
    );

    return (
        <Modal
            isOpen={isOpen}
            onClose={onClose}
            // UPDATED: Generic title
            title="Edit Media Details"
            footer={modalFooter}
        >
            <div className="space-y-4">
                <Input
                    label="Filename"
                    type="text"
                    value={filename}
                    onChange={(e) => setFilename(e.target.value)}
                    placeholder="Enter a new filename"
                />
                <Input
                    as="textarea"
                    label="Description"
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    placeholder="Enter a description"
                    rows={4}
                />
            </div>
        </Modal>
    );
};