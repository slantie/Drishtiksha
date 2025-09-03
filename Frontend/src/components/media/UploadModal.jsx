// src/components/media/UploadModal.jsx

import React, { useState, useRef, useCallback } from "react";
import {
  Upload,
  CheckCircle,
  FileVideo,
  FileAudio,
  FileImage,
  XCircle,
} from "lucide-react"; // Added XCircle for error state
import { Button } from "../ui/Button.jsx";
import { Input } from "../ui/Input.jsx";
import { Modal } from "../ui/Modal.jsx";
import { showToast } from "../../utils/toast.js";
import { useUploadMediaMutation } from "../../hooks/useMediaQuery.jsx";
import { config } from "../../config/env.js"; // Import config for max file size

// NOTE: Ensure these MIME types are consistent with your backend's multer middleware
const ALLOWED_MIME_TYPES = [
  "video/mp4",
  "video/webm",
  "video/quicktime",
  "video/x-msvideo", // Added x-msvideo
  "audio/mpeg",
  "audio/wav",
  "audio/ogg",
  "audio/mp4", // Added audio/mp4
  "image/jpeg",
  "image/png",
  "image/webp",
];

// Max file size in MB from environment config, convert to bytes for comparison
const MAX_FILE_SIZE_BYTES = (config.VITE_MAX_FILE_SIZE || 150) * 1024 * 1024; // Default to 150MB if not in config

const getFileIcon = (fileType) => {
  if (fileType?.startsWith("video/"))
    return <FileVideo className="w-12 h-12" />;
  if (fileType?.startsWith("audio/"))
    return <FileAudio className="w-12 h-12" />;
  if (fileType?.startsWith("image/"))
    return <FileImage className="w-12 h-12" />;
  return <Upload className="w-12 h-12" />; // Default icon
};

export const UploadModal = ({ isOpen, onClose }) => {
  const [mediaFile, setMediaFile] = useState(null);
  const [description, setDescription] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const [fileError, setFileError] = useState(null); // New state for file-specific errors
  const fileInputRef = useRef(null);
  const uploadMutation = useUploadMediaMutation();

  const clearForm = useCallback(() => {
    setMediaFile(null);
    setDescription("");
    setFileError(null); // Clear file error
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, []);

  const validateAndSetFile = (file) => {
    setFileError(null); // Reset error
    if (!file) {
      clearForm();
      return;
    }

    if (!ALLOWED_MIME_TYPES.includes(file.type)) {
      setFileError(
        `Unsupported file type: ${file.type}. Please select a valid video, audio, or image file.`
      );
      setMediaFile(null);
      showToast.error("Unsupported file type.");
      return;
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      setFileError(
        `File size exceeds limit (${(
          MAX_FILE_SIZE_BYTES /
          (1024 * 1024)
        ).toFixed(0)}MB).`
      );
      setMediaFile(null);
      showToast.error("File size too large.");
      return;
    }

    setMediaFile(file);
  };

  const handleFileChange = (e) => {
    validateAndSetFile(e.target.files[0]);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    validateAndSetFile(e.dataTransfer.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!mediaFile) {
      showToast.warning("Please select a file to upload.");
      return;
    }
    if (fileError) {
      // Prevent submission if there's a file error
      showToast.error(fileError);
      return;
    }

    const formData = new FormData();
    formData.append("media", mediaFile); // Field name MUST match backend multer config
    formData.append("description", description);

    await uploadMutation.mutateAsync(formData);

    // The onSuccess handler in the hook takes care of navigation and toasts.
    clearForm();
    onClose();
  };

  const modalFooter = (
    <Button
      onClick={handleSubmit}
      isLoading={uploadMutation.isPending}
      disabled={!mediaFile || !!fileError || uploadMutation.isPending}
      className="w-full"
      size="lg"
    >
      {!uploadMutation.isPending && <Upload className="mr-2 h-4 w-4" />}
      {uploadMutation.isPending
        ? "Uploading & Analyzing..."
        : "Upload & Analyze"}
    </Button>
  );

  return (
    <Modal
      isOpen={isOpen}
      onClose={() => {
        onClose();
        clearForm();
      }} // Clear form on modal close
      title="Upload Media"
      description="Select a video, audio, or image file to begin analysis."
      footer={modalFooter}
    >
      <form onSubmit={handleSubmit} className="space-y-4">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
            dragActive
              ? "border-primary-main bg-primary-main/10"
              : "border-light-secondary dark:border-dark-secondary"
          } ${fileError ? "border-red-500 bg-red-500/5" : ""}`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept={ALLOWED_MIME_TYPES.join(",")} // Use allowed mime types for browser hint
            onChange={handleFileChange}
            className="hidden"
          />
          {fileError ? (
            <div className="space-y-2 flex flex-col items-center text-red-500">
              <XCircle className="w-12 h-12" />
              <p className="font-semibold">{fileError}</p>
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
          ) : mediaFile ? (
            <div className="space-y-2 flex flex-col items-center">
              <CheckCircle className="w-12 h-12 text-green-500" />
              <p className="font-semibold">{mediaFile.name}</p>
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
              {getFileIcon(null)} {/* Pass null to get default upload icon */}
              <p>
                <span className="font-semibold text-primary-main">
                  Click to browse
                </span>{" "}
                or drag & drop
              </p>
              <p className="text-xs">
                Supports Video, Audio, and Image. Max{" "}
                {(MAX_FILE_SIZE_BYTES / (1024 * 1024)).toFixed(0)}MB.
              </p>
            </div>
          )}
        </div>
        <Input
          as="textarea"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Add a description (optional)..."
          rows={3}
          disabled={uploadMutation.isPending}
        />
      </form>
    </Modal>
  );
};
