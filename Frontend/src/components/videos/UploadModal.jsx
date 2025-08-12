// src/components/videos/UploadModal.jsx

import React, { useState, useRef, useCallback } from "react";
import { Upload, X, CheckCircle, FileVideo, Loader2 } from "lucide-react";
import { Button } from "../ui/Button";
import { showToast } from "../../utils/toast";

export const UploadModal = ({ isOpen, onClose, onUpload }) => {
    const [videoFile, setVideoFile] = useState(null);
    const [description, setDescription] = useState("");
    const [isUploading, setIsUploading] = useState(false);
    const [dragActive, setDragActive] = useState(false);
    const fileInputRef = useRef(null);

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
        if (!videoFile) {
            return showToast.warning("Please select a video file.");
        }
        setIsUploading(true);
        const formData = new FormData();
        formData.append("video", videoFile);
        formData.append("description", description);
        try {
            await onUpload(formData);
            clearForm();
            onClose();
        } catch (error) {
            console.error("Upload failed:", error);
        } finally {
            setIsUploading(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-[-25px] z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
            <div className="relative w-full max-w-2xl bg-light-background dark:bg-dark-background rounded-3xl p-8 shadow-2xl border border-light-secondary dark:border-dark-secondary">
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 p-2 rounded-full hover:bg-light-muted-background dark:hover:bg-dark-muted-background"
                >
                    <X />
                </button>
                <h2 className="text-2xl font-bold mb-6">Upload Video</h2>
                <form onSubmit={handleSubmit} className="space-y-6">
                    {/* Drag and drop area */}
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
                        className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-colors ${
                            dragActive
                                ? "border-primary-main"
                                : "border-light-secondary dark:border-dark-secondary"
                        }`}
                    >
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept="video/*"
                            onChange={(e) =>
                                handleFileSelect(e.target.files[0])
                            }
                            className="hidden"
                        />
                        {videoFile ? (
                            <div className="space-y-2">
                                <CheckCircle className="w-12 h-12 text-green-500 mx-auto" />
                                <p>{videoFile.name}</p>
                                <button
                                    type="button"
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        clearForm();
                                    }}
                                    className="text-primary-main"
                                >
                                    Choose another file
                                </button>
                            </div>
                        ) : (
                            <div className="space-y-2">
                                <FileVideo className="w-12 h-12 mx-auto" />
                                <p>Drag & drop or click to browse</p>
                            </div>
                        )}
                    </div>
                    <textarea
                        value={description}
                        onChange={(e) => setDescription(e.target.value)}
                        placeholder="Description (optional)"
                        className="w-full p-3 bg-light-muted-background dark:bg-dark-muted-background rounded-lg border border-light-secondary dark:border-dark-secondary"
                        rows={3}
                    ></textarea>
                    <Button
                        type="submit"
                        disabled={isUploading || !videoFile}
                        className="w-full py-3 text-lg"
                    >
                        {isUploading ? (
                            <>
                                <Loader2 className="animate-spin mr-2" />{" "}
                                Uploading...
                            </>
                        ) : (
                            <>
                                {" "}
                                <Upload className="mr-2" /> Upload & Analyze
                            </>
                        )}
                    </Button>
                </form>
            </div>
        </div>
    );
};
