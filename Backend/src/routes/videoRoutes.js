import express from "express";
import multer from "multer";
import { PrismaClient } from "@prisma/client";
import { authenticateToken } from "../middleware/auth.js";
import {
    processVideoForDeepfake,
    getAnalysisResults,
    getAllAnalyses,
} from "../controllers/deepfakeController.js";
import path from "path";

const router = express.Router();
const prisma = new PrismaClient();

// Enhanced Multer setup with file validation
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, "uploads/videos/");
    },
    filename: function (req, file, cb) {
        const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
        cb(
            null,
            file.fieldname +
                "-" +
                uniqueSuffix +
                path.extname(file.originalname)
        );
    },
});

const fileFilter = (req, file, cb) => {
    // Check file type
    if (file.mimetype.startsWith("video/")) {
        cb(null, true);
    } else {
        cb(new Error("Only video files are allowed!"), false);
    }
};

const upload = multer({
    storage: storage,
    fileFilter: fileFilter,
    limits: {
        fileSize: 100 * 1024 * 1024, // 100MB limit
    },
});

// Video upload with enhanced validation
router.post(
    "/upload",
    authenticateToken,
    upload.single("video"),
    async (req, res) => {
        try {
            const { description } = req.body;

            if (!req.file) {
                return res.status(400).json({
                    success: false,
                    message: "No video file uploaded",
                });
            }

            // Validate file size and type
            const allowedTypes = [
                "video/mp4",
                "video/avi",
                "video/mov",
                "video/wmv",
            ];
            if (!allowedTypes.includes(req.file.mimetype)) {
                return res.status(400).json({
                    success: false,
                    message:
                        "Unsupported video format. Please upload MP4, AVI, MOV, or WMV files.",
                });
            }

            // Create video record in DB
            const video = await prisma.video.create({
                data: {
                    filename: req.file.originalname,
                    filepath: req.file.path,
                    mimetype: req.file.mimetype,
                    size: req.file.size,
                    description,
                    status: "UPLOADED",
                    userId: req.user.id,
                },
            });

            res.status(201).json({
                success: true,
                data: video,
                message: "Video uploaded successfully! Ready for analysis.",
            });
        } catch (error) {
            console.error("Error uploading video:", error);
            res.status(500).json({
                success: false,
                message: "Failed to upload video",
            });
        }
    }
);

// Analyze video for deepfakes
router.post("/:videoId/analyze", authenticateToken, processVideoForDeepfake);

// Get analysis results
router.get("/:videoId/analysis", authenticateToken, getAnalysisResults);

// Get all analyses with pagination
router.get("/analyses", authenticateToken, getAllAnalyses);

// GET /api/video
router.get("/", async (req, res) => {
    try {
        const whereClause =
            req.user?.role === "ADMIN" ? {} : { userId: req.user?.id };

        const videos = await prisma.video.findMany({
            where: whereClause,
            include: {
                user: {
                    select: {
                        id: true,
                        firstName: true,
                        lastName: true,
                        email: true,
                    },
                },
                analysis: true,
            },
            orderBy: { createdAt: "desc" },
        });

        res.json({ success: true, data: videos });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to fetch videos",
        });
    }
});

// GET /api/video/:id
router.get("/:id", authenticateToken, async (req, res) => {
    try {
        const video = await prisma.video.findUnique({
            where: { id: req.params.id },
            include: {
                user: {
                    select: {
                        id: true,
                        firstName: true,
                        lastName: true,
                        email: true,
                    },
                },
                analysis: true,
            },
        });

        if (!video) {
            return res.status(404).json({
                success: false,
                message: "Video not found",
            });
        }

        // Check permissions
        if (video.userId !== req.user.id && req.user.role !== "ADMIN") {
            return res.status(403).json({
                success: false,
                message: "Access denied",
            });
        }

        res.json({ success: true, data: video });
    } catch (error) {
        console.error(error);
        res.status(500).json({
            success: false,
            message: "Failed to fetch video",
        });
    }
});

export default router;
