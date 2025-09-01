// src/components/media/UploadModal.jsx

import React, { useState, useRef, useCallback } from "react";
import { Upload, CheckCircle, FileVideo, FileAudio, FileImage } from "lucide-react";
import { Button } from "../ui/Button.jsx";
import { Input } from "../ui/Input.jsx";
import { Modal } from "../ui/Modal.jsx";
import { showToast } from "../../utils/toast.js";
import { useUploadMediaMutation } from "../../hooks/useMediaQuery.jsx";

const ALLOWED_MIME_TYPES = [
    "video/mp4", "video/webm", "video/quicktime",
    "audio/mpeg", "audio/wav", "audio/ogg",
    "image/jpeg", "image/png", "image/webp",
];

const getFileIcon = (fileType) => {
    if (fileType?.startsWith("video/")) return <FileVideo className="w-12 h-12" />;
    if (fileType?.startsWith("audio/")) return <FileAudio className="w-12 h-12" />;
    if (fileType?.startsWith("image/")) return <FileImage className="w-12 h-12" />;
    return <Upload className="w-12 h-12" />; // Default icon
};

export const UploadModal = ({ isOpen, onClose }) => {
    const [mediaFile, setMediaFile] = useState(null);
    const [description, setDescription] = useState("");
    const [dragActive, setDragActive] = useState(false);
    const fileInputRef = useRef(null);
    const uploadMutation = useUploadMediaMutation();

    const clearForm = useCallback(() => {
        setMediaFile(null);
        setDescription("");
        if (fileInputRef.current) fileInputRef.current.value = "";
    }, []);

    const handleFileSelect = (file) => {
        if (file && ALLOWED_MIME_TYPES.includes(file.type)) {
            setMediaFile(file);
        } else if (file) {
            showToast.error("Unsupported file type. Please select a valid video, audio, or image file.");
            clearForm();
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!mediaFile) return showToast.warning("Please select a file to upload.");

        const formData = new FormData();
        // FIX: The field name must match the backend multer middleware ('media').
        formData.append("media", mediaFile);
        formData.append("description", description);

        await uploadMutation.mutateAsync(formData);

        // The onSuccess handler in the hook takes care of navigation and toasts.
        clearForm();
        onClose();
    };

    const modalFooter = (
        <Button onClick={handleSubmit} isLoading={uploadMutation.isPending} disabled={!mediaFile} className="w-full" size="lg">
            {!uploadMutation.isPending && <Upload className="mr-2 h-4 w-4" />}
            Upload & Analyze
        </Button>
    );

    return (
        <Modal
            isOpen={isOpen}
            onClose={onClose}
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
                        dragActive ? "border-primary-main bg-primary-main/10" : "border-light-secondary dark:border-dark-secondary"
                    }`}
                >
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept="video/*,audio/*,image/*"
                        onChange={(e) => handleFileSelect(e.target.files[0])}
                        className="hidden"
                    />
                    {mediaFile ? (
                        <div className="space-y-2 flex flex-col items-center">
                            <CheckCircle className="w-12 h-12 text-green-500" />
                            <p className="font-semibold">{mediaFile.name}</p>
                            <button type="button" onClick={(e) => { e.stopPropagation(); clearForm(); }} className="text-sm text-primary-main font-semibold hover:underline">
                                Choose another file
                            </button>
                        </div>
                    ) : (
                        <div className="space-y-2 flex flex-col items-center text-light-muted-text dark:text-dark-muted-text">
                            {getFileIcon()}
                            <p>
                                <span className="font-semibold text-primary-main">
                                    Click to browse
                                </span> or drag & drop
                            </p>
                            <p className="text-xs">Supports Video, Audio, and Image. Max 150MB.</p>
                        </div>
                    )}
                </div>
                <Input
                    as="textarea"
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    placeholder="Add a description (optional)..."
                    rows={3}
                />
            </form>
        </Modal>
    );
};