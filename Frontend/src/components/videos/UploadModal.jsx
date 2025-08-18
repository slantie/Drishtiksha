// src/components/videos/UploadModal.jsx

import React, { useState, useRef, useCallback } from "react";
import { Upload, CheckCircle, FileVideo } from "lucide-react";
import { Button } from "../ui/Button";
import { Input } from "../ui/Input";
import { Modal } from "../ui/Modal";
import { showToast } from "../../utils/toast";
import { useUploadVideoMutation } from "../../hooks/useVideosQuery.jsx";

export const UploadModal = ({ isOpen, onClose }) => {
    const [videoFile, setVideoFile] = useState(null);
    const [description, setDescription] = useState("");
    const [dragActive, setDragActive] = useState(false);
    const fileInputRef = useRef(null);
    const uploadMutation = useUploadVideoMutation();

    const clearForm = useCallback(() => {
        setVideoFile(null);
        setDescription("");
        if (fileInputRef.current) fileInputRef.current.value = "";
    }, []);

    const handleFileSelect = (file) => {
        if (file && file.type.startsWith("video/")) {
            setVideoFile(file);
        } else {
            showToast.error("Please select a valid video file.");
            clearForm();
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!videoFile) return showToast.warning("Please select a video file.");

        const formData = new FormData();
        formData.append("video", videoFile);
        formData.append("description", description);

        // REFACTORED: The mutation now handles navigation and toast messages.
        await uploadMutation.mutateAsync(formData);

        clearForm();
        onClose();
    };

    const modalFooter = (
        <Button
            onClick={handleSubmit}
            isLoading={uploadMutation.isPending}
            disabled={!videoFile}
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
            title="Upload Video"
            description="Select a video file to begin analysis."
            footer={modalFooter}
        >
            <form onSubmit={handleSubmit} className="space-y-4">
                <div
                    onDragOver={(e) => {
                        e.preventDefault();
                        setDragActive(true);
                    }}
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
                        accept="video/*"
                        onChange={(e) => handleFileSelect(e.target.files[0])}
                        className="hidden"
                    />
                    {videoFile ? (
                        <div className="space-y-2 flex flex-col items-center">
                            <CheckCircle className="w-12 h-12 text-green-500" />
                            <p className="font-semibold">{videoFile.name}</p>
                            <button
                                type="button"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    clearForm();
                                }}
                                className="text-sm text-primary-main font-semibold hover:underline"
                            >
                                Choose another file
                            </button>
                        </div>
                    ) : (
                        <div className="space-y-2 flex flex-col items-center text-light-muted-text dark:text-dark-muted-text">
                            <FileVideo className="w-12 h-12" />
                            <p>
                                <span className="font-semibold text-primary-main">
                                    Click to browse
                                </span>{" "}
                                or drag & drop
                            </p>
                            <p className="text-xs">Maximum file size: 100MB</p>
                        </div>
                    )}
                </div>
                {/* REFACTOR: Replaced textarea with our consistent Input component. */}
                <Input
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    placeholder="Description (optional)"
                />
            </form>
        </Modal>
    );
};
