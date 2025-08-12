/**
 * @file src/controllers/videoController.js
 * @description Controller functions for managing video CRUD operations.
 */

import { PrismaClient } from "@prisma/client";
import { v2 as cloudinary } from "cloudinary";
import fs from "fs/promises";
import { videoProcessorQueue } from "../queue/videoProcessorQueue.js";
import path from "path";

const prisma = new PrismaClient();

//==================================================================
// Cloudinary Configuration
//==================================================================
cloudinary.config({
    cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
    api_key: process.env.CLOUDINARY_API_KEY,
    api_secret: process.env.CLOUDINARY_API_SECRET,
});

//==================================================================
// Controller for handling video upload
//==================================================================
export const uploadVideo = async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ success: false, message: "No video file provided" });
        }

        const { description } = req.body;
        const userId = req.user.id;
        let videoCloudinaryResult = null; // A variable to hold the Cloudinary result

        try {
            // Upload video to Cloudinary
            videoCloudinaryResult = await cloudinary.uploader.upload(req.file.path, {
                resource_type: "video",
                folder: "deepfake-analysis-videos",
                // The public_id will be derived from the original filename without the extension
                public_id: path.parse(req.file.originalname).name, 
            });

            // Delete the temporary file from the local server after a successful upload
            await fs.unlink(req.file.path).catch(err => {
                console.error(`Failed to delete local file ${req.file.path}:`, err);
            });
        } catch (uploadError) {
            console.error("Cloudinary upload or file deletion error:", uploadError);
            return res.status(500).json({ success: false, message: "Failed to upload video to Cloudinary" });
        }

        // Create a new video record in the database with Cloudinary URL and public_id
        const newVideo = await prisma.video.create({
            data: {
                filename: req.file.originalname, // Store original filename
                mimetype: req.file.mimetype,
                size: req.file.size,
                description: description || "",
                url: videoCloudinaryResult.secure_url,
                publicId: videoCloudinaryResult.public_id,
                userId: userId,
                status: "UPLOADED",
            },
        });

        // ==========================================================
        // CHANGE: Correctly call the videoProcessorQueue function.
        // It's a function, not an object with an 'add' method.
        // ==========================================================
        videoProcessorQueue({
            videoId: newVideo.id,
            videoUrl: newVideo.url,
            userId: newVideo.userId,
        });

        console.log(`Video ${newVideo.id} added to in-memory processing queue.`);

        return res.status(201).json({
            success: true,
            message: "Video uploaded and queued for processing successfully.",
            data: newVideo,
        });
    } catch (error) {
        console.error("Video upload error:", error);
        // The file is already deleted in the try block, so no need to delete it again here.
        return res.status(500).json({ success: false, message: "Failed to upload video" });
    }
};

//==================================================================
// Controller for fetching all videos for the current user or all for an admin
//==================================================================
export const getAllVideos = async (req, res) => {
    try {
        const whereClause = req.user?.role === "ADMIN" ? {} : { userId: req.user?.id };
        const videos = await prisma.video.findMany({
            where: whereClause,
            include: {
                user: { select: { id: true, firstName: true, lastName: true, email: true } },
                analyses: true,
            },
            orderBy: { createdAt: "desc" },
        });
        // The frontend will now use the video.url property for playback
        return res.json({ success: true, data: videos });
    } catch (error) {
        console.error("Error fetching all videos:", error);
        return res.status(500).json({ success: false, message: "Failed to fetch videos" });
    }
};

//==================================================================
// Controller for fetching a single video by ID
//==================================================================
export const getVideoById = async (req, res) => {
    try {
        const video = await prisma.video.findUnique({
            where: { id: req.params.id },
            include: {
                user: { select: { id: true, firstName: true, lastName: true, email: true } },
                analyses: true,
            },
        });

        if (!video) {
            return res.status(404).json({ success: false, message: "Video not found" });
        }
        if (video.userId !== req.user.id && req.user.role !== "ADMIN") {
            return res.status(403).json({ success: false, message: "Access denied" });
        }
        // The frontend will now use the video.url property for playback
        return res.json({ success: true, data: video });
    } catch (error) {
        console.error("Error fetching video by ID:", error);
        return res.status(500).json({ success: false, message: "Failed to fetch video" });
    }
};

//==================================================================
// Controller for updating a video's description
//==================================================================
export const updateVideo = async (req, res) => {
    try {
        const { id } = req.params;
        const { description, filename } = req.body;

        if (!description && !filename) {
            return res.status(400).json({ success: false, message: "No data provided to update." });
        }

        const video = await prisma.video.findUnique({ where: { id } });

        if (!video) {
            return res.status(404).json({ success: false, message: "Video not found" });
        }

        if (video.userId !== req.user.id && req.user.role !== "ADMIN") {
            return res.status(403).json({ success: false, message: "Access denied" });
        }

        const updateData = {};
        if (description !== undefined) updateData.description = description;
        if (filename !== undefined) updateData.filename = filename;

        const updatedVideo = await prisma.video.update({
            where: { id },
            data: updateData,
        });

        return res.json({ success: true, data: updatedVideo });
    } catch (error) {
        console.error("Error updating video:", error);
        return res.status(500).json({ success: false, message: "Failed to update video" });
    }
};

//==================================================================
// Controller for deleting a video and its file from Cloudinary
//==================================================================
export const deleteVideo = async (req, res) => {
    try {
        const { id } = req.params;

        const video = await prisma.video.findUnique({ where: { id } });

        if (!video) {
            return res.status(404).json({ success: false, message: "Video not found" });
        }

        if (video.userId !== req.user.id && req.user.role !== "ADMIN") {
            return res.status(403).json({ success: false, message: "Access denied" });
        }
        
        // Delete the video from Cloudinary using its publicId
        if (video.publicId) {
            await cloudinary.uploader.destroy(video.publicId, {
                resource_type: "video"
            });
        }

        await prisma.video.delete({ where: { id } });

        return res.json({ success: true, message: "Video deleted successfully" });
    } catch (error) {
        console.error("Error deleting video:", error);
        return res.status(500).json({ success: false, message: "Failed to delete video" });
    }
};
