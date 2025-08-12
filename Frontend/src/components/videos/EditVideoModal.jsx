// src/components/videos/EditVideoModal.jsx

import React, { useState, useEffect } from "react";
import { Save, X, Loader2 } from "lucide-react";
import { Button } from "../ui/Button";

export const EditVideoModal = ({ isOpen, onClose, video, onUpdate }) => {
    const [description, setDescription] = useState("");
    const [filename, setFilename] = useState("");
    const [isSaving, setIsSaving] = useState(false);

    useEffect(() => {
        if (video) {
            setDescription(video.description || "");
            setFilename(video.filename || "");
        }
    }, [video]);

    const handleSave = async () => {
        setIsSaving(true);
        try {
            await onUpdate(video.id, { description, filename });
            onClose();
        } catch (error) {
            console.error("Update failed:", error);
        } finally {
            setIsSaving(false);
        }
    };

    if (!isOpen || !video) return null;

    return (
        <div className="fixed inset-[-25px] z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
            <div className="relative w-full max-w-lg bg-light-background dark:bg-dark-background rounded-3xl p-8 shadow-2xl border border-light-secondary dark:border-dark-secondary">
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 p-2 rounded-full hover:bg-light-muted-background dark:hover:bg-dark-muted-background"
                >
                    <X />
                </button>
                <h2 className="text-2xl font-bold mb-6">Edit Video</h2>
                <div className="space-y-4">
                    <input
                        type="text"
                        value={filename}
                        onChange={(e) => setFilename(e.target.value)}
                        placeholder="Filename"
                        className="w-full p-3 bg-light-muted-background dark:bg-dark-muted-background rounded-lg border border-light-secondary dark:border-dark-secondary"
                    />
                    <textarea
                        value={description}
                        onChange={(e) => setDescription(e.target.value)}
                        placeholder="Description"
                        className="w-full p-3 bg-light-muted-background dark:bg-dark-muted-background rounded-lg border border-light-secondary dark:border-dark-secondary"
                        rows={5}
                    ></textarea>
                    <Button
                        onClick={handleSave}
                        disabled={isSaving}
                        className="w-full py-3 text-lg"
                    >
                        {isSaving ? (
                            <>
                                <Loader2 className="animate-spin mr-2" />{" "}
                                Saving...
                            </>
                        ) : (
                            <>
                                <Save className="mr-2" /> Save Changes
                            </>
                        )}
                    </Button>
                </div>
            </div>
        </div>
    );
};
