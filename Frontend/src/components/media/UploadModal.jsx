// src/components/media/UploadModal.jsx

import React, { useState, useRef, useCallback } from "react";
import { Upload, CheckCircle, FileVideo, FileAudio, FileImage } from "lucide-react";
import { Button } from "../ui/Button.jsx";
import { Input } from "../ui/Input.jsx";
import { Modal } from "../ui/Modal.jsx";
import { showToast } from "../../utils/toast.js";
// UPDATED: Import the new generic mutation hook
import { useUploadMediaMutation } from "../../hooks/useMediaQuery.jsx";

// Define allowed MIME types for frontend validation
const ALLOWED_MIME_TYPES = [
    "video/mp4", "video/webm", "video/quicktime",
    "audio/mpeg", "audio/wav", "audio/ogg",
    "image/jpeg", "image/png", "image/webp",
];

// Helper to get the correct icon for a given file type
const getFileIcon = (fileType) => {
    if (fileType?.startsWith("video/")) return <FileVideo className="w-12 h-12" />;
    if (fileType?.startsWith("audio/")) return <FileAudio className="w-12 h-12" />;
    if (fileType?.startsWith("image/")) return <FileImage className="w-12 h-12" />;
    return <FileVideo className="w-12 h-12" />; // Default icon
};

export const UploadModal = ({ isOpen, onClose }) => {
    // RENAMED: from videoFile to mediaFile
    const [mediaFile, setMediaFile] = useState(null);
    const [description, setDescription] = useState("");
    const [dragActive, setDragActive] = useState(false);
    const fileInputRef = useRef(null);
    // UPDATED: Using the new generic mutation hook
    const uploadMutation = useUploadMediaMutation();

    const clearForm = useCallback(() => {
        setMediaFile(null);
        setDescription("");
        if (fileInputRef.current) fileInputRef.current.value = "";
    }, []);

    const handleFileSelect = (file) => {
        // UPDATED: Validate against a list of allowed MIME types
        if (file && ALLOWED_MIME_TYPES.includes(file.type)) {
            setMediaFile(file);
        } else {
            showToast.error("Unsupported file type. Please select a valid video, audio, or image file.");
            clearForm();
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!mediaFile) return showToast.warning("Please select a file to upload.");

        const formData = new FormData();
        // CRITICAL UPDATE: The field name must be "file" to match the backend
        formData.append("file", mediaFile);
        formData.append("description", description);

        await uploadMutation.mutateAsync(formData);

        // onSuccess is now handled within the hook, which includes navigation.
        // We just need to clear the form and close the modal.
        clearForm();
        onClose();
    };

    const modalFooter = (
        <Button
            onClick={handleSubmit}
            isLoading={uploadMutation.isPending}
            disabled={!mediaFile}
            className="w-full"
            size="lg"
        >
            {!uploadMutation.isPending && <Upload className="mr-2 h-4 w-4" />}
            Upload & Analyze
        </Button>
    );

    return (
        <Modal
            isOpen={isOpen}
            onClose={onClose}
            // UPDATED: Generic title and description
            title="Upload Media"
            description="Select a video, audio, or image file to begin analysis."
            footer={modalFooter}
        >
            <form onSubmit={handleSubmit} className="space-y-4">
                <div
                    onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
                    onDragLeave={() => setDragActive(false)}
                    onDrop={(e) => {
                        e.preventDefault();
                        setDragActive(false);
                        handleFileSelect(e.dataTransfer.files[0]);
                    }}
                    onClick={() => fileInputRef.current?.click()}
                    className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                        dragActive
                            ? "border-primary-main bg-primary-main/10"
                            : "border-light-secondary dark:border-dark-secondary"
                    }`}
                >
                    <input
                        ref={fileInputRef}
                        type="file"
                        // UPDATED: Accept multiple media types
                        accept="video/*,audio/*,image/*"
                        onChange={(e) => handleFileSelect(e.target.files[0])}
                        className="hidden"
                    />
                    {mediaFile ? (
                        <div className="space-y-2 flex flex-col items-center">
                            <CheckCircle className="w-12 h-12 text-green-500" />
                            <p className="font-semibold">{mediaFile.name}</p>
                            <button
                                type="button"
                                onClick={(e) => { e.stopPropagation(); clearForm(); }}
                                className="text-sm text-primary-main font-semibold hover:underline"
                            >
                                Choose another file
                            </button>
                        </div>
                    ) : (
                        <div className="space-y-2 flex flex-col items-center text-light-muted-text dark:text-dark-muted-text">
                            {/* UPDATED: Dynamic icon based on file type */}
                            {getFileIcon(mediaFile?.type)}
                            <p>
                                <span className="font-semibold text-primary-main">
                                    Click to browse
                                </span>{" "}
                                or drag & drop
                            </p>
                            <p className="text-xs">Supports Video, Audio, and Image files. Max 100MB.</p>
                        </div>
                    )}
                </div>
                <Input
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    placeholder="Description (optional)"
                />
            </form>
        </Modal>
    );
};