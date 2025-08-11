import { prisma } from "../config/database.js";
import path from "path";
import fs from "fs";

export const uploadVideo = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: "No video file uploaded",
      });
    }

    const { description } = req.body;

    // Save video record in DB
    const video = await prisma.video.create({
      data: {
        filename: req.file.filename,
        filepath: `/uploads/videos/${req.file.filename}`,
        mimetype: req.file.mimetype,
        size: req.file.size,
        description: description || null,
        userId: req.user.id, // from auth middleware
      },
    });

    res.status(201).json({
      success: true,
      message: "Video uploaded successfully",
      video,
    });
  } catch (error) {
    console.error("Video upload error:", error);
    res.status(500).json({
      success: false,
      message: "Internal server error",
    });
  }
};
