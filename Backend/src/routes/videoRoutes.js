import express from "express";
import multer from "multer";
import { PrismaClient } from "@prisma/client";
import { authenticateToken } from "../middleware/auth.js";

const router = express.Router();
const prisma = new PrismaClient();

// Multer setup — temporary local storage
const upload = multer({ dest: "uploads/" });

// --- FIX #3: Add authMiddleware to the route ---
// This will protect the route and add `req.user` from the token.
router.post("/upload", authenticateToken, upload.single("video"), async (req, res) => {
  try {
    // Now we get the description from the request body
    const { description } = req.body;

    if (!req.file) {
      return res.status(400).json({ success: false, message: "No file uploaded" });
    }

    // Create video record in DB using the secure userId from the token
    const video = await prisma.video.create({
      data: {
        filename: req.file.originalname,
        filepath: req.file.path,
        mimetype: req.file.mimetype,
        size: req.file.size,
        description,
        userId: req.user.id,
      },
    });

    res.status(201).json({ success: true, data: video, message: "Video uploaded successfully!" });

    // Video to be sent to server for further processing.
  } catch (error) {
    console.error("Error uploading video:", error);
    res.status(500).json({ success: false, message: "Failed to upload video" });
  }
});

// GET /api/video
router.get("/", async (req, res) => {
  try {
    const videos = await prisma.video.findMany({ include: { user: true } });
    res.json({ success: true, data: videos });
  } catch (error) {
    console.error(error);
    res.status(500).json({ success: false, message: "Failed to fetch videos" });
  }
});

// GET /api/video/:id
router.get("/:id", async (req, res) => {
  try {
    const video = await prisma.video.findUnique({
      where: { id: req.params.id },
      include: { user: true },
    });
    if (!video) return res.status(404).json({ success: false, message: "Video not found" });
    res.json({ success: true, data: video });
  } catch (error){
    console.error(error);
    res.status(500).json({ success: false, message: "Failed to fetch video" });
  }
});

export default router;