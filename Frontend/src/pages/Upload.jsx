import React, { useState } from "react";
import { showToast } from "../utils/toast"; // Using your app's toast utility

export const UploadPage = () => {
  const [videoFile, setVideoFile] = useState(null);
  const [description, setDescription] = useState("");
  const [isUploading, setIsUploading] = useState(false);

  // Use the same auth token retrieval logic as your other components
  const getAuthToken = () => {
    return localStorage.getItem("authToken") || sessionStorage.getItem("authToken");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // --- FIX #1: Use the correct function to get the token ---
    const token = getAuthToken();
    if (!token) {
      showToast.error("You must be logged in to upload videos.");
      return;
    }

    if (!videoFile) {
      showToast.warning("Please select a video file to upload.");
      return;
    }

    setIsUploading(true);
    const loadingToast = showToast.loading("Uploading video...");

    const formData = new FormData();
    // --- FIX #2: Use the correct state variable `videoFile` ---
    formData.append("video", videoFile);
    formData.append("description", description);

    try {
      const res = await fetch("http://localhost:4000/api/video/upload", {
        method: "POST",
        headers: {
          // No 'Content-Type' needed; the browser sets it for FormData
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      const data = await res.json();
      showToast.dismiss(loadingToast);

      if (res.ok && data.success) {
        showToast.success("Video uploaded successfully!");
        // Clear the form
        setVideoFile(null);
        setDescription("");
        // Reset file input visually
        document.querySelector('input[type="file"]').value = "";
      } else {
        showToast.error(data.message || "Upload failed. Please try again.");
      }
    } catch (err) {
      showToast.dismiss(loadingToast);
      console.error(err);
      showToast.error("An error occurred while uploading.");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="max-w-lg mx-auto p-8 space-y-6 bg-light-background dark:bg-dark-background border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-lg shadow-lg">
      <div className="text-center">
        <h1 className="text-2xl font-bold text-light-text dark:text-dark-text">Upload Video</h1>
        <p className="text-light-muted-text dark:text-dark-muted-text">Share your content with the world</p>
      </div>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-light-text dark:text-dark-text mb-1">Video File</label>
          <input
            type="file"
            accept="video/*"
            onChange={(e) => setVideoFile(e.target.files[0])}
            className="block w-full text-sm text-light-muted-text dark:text-dark-muted-text
              file:mr-4 file:py-2 file:px-4
              file:rounded-lg file:border-0
              file:text-sm file:font-semibold
              file:bg-light-highlight/10 dark:file:bg-dark-highlight/10
              file:text-light-highlight dark:file:text-dark-highlight
              hover:file:bg-light-highlight/20 dark:hover:file:bg-dark-highlight/20
              cursor-pointer"
            disabled={isUploading}
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-light-text dark:text-dark-text mb-1">Description (Optional)</label>
          <textarea
            placeholder="Describe your video..."
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            className="w-full px-3 py-2 bg-light-muted-background dark:bg-dark-muted-background border border-light-muted-text/20 dark:border-dark-muted-text/20 rounded-lg text-light-text dark:text-dark-text focus:outline-none focus:border-light-highlight dark:focus:border-dark-highlight transition-colors"
            rows={4}
            disabled={isUploading}
          />
        </div>
        <button
          type="submit"
          disabled={isUploading || !videoFile}
          className="w-full bg-gradient-to-r from-light-highlight to-light-highlight/90 dark:from-dark-highlight dark:to-dark-highlight/90 text-white font-bold py-3 px-4 rounded-lg shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 flex items-center justify-center"
        >
          {isUploading ? "Uploading..." : "Upload"}
        </button>
      </form>
    </div>
  );
};