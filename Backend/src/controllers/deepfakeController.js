import { prisma } from "../config/database.js";
import path from "path";
import fs from "fs";

// Mock analysis function remains the same
const generateMockAnalysis = (filename, fileSize, model) => {
    const isLikelyReal = Math.random() > 0.3;
    const baseConfidence = isLikelyReal ? 0.75 + Math.random() * 0.2 : 0.6 + Math.random() * 0.3;
    const confidence = Math.min(0.98, Math.max(0.52, baseConfidence + (Math.random() - 0.5) * 0.1));
    const baseProcTime = 2.5 + (fileSize / (1024 * 1024)) * 0.1;
    const processingTime = baseProcTime + (Math.random() - 0.5) * 1.0;
    const prediction = confidence > 0.75 ? "REAL" : "FAKE";

    return {
        prediction,
        confidence: parseFloat(confidence.toFixed(4)),
        processing_time: parseFloat(Math.max(1.2, processingTime).toFixed(2)),
        model_version: `${model}-mock`,
        analysis_metadata: {
            frames_analyzed: Math.floor(20 + Math.random() * 40),
            video_duration: parseFloat((5 + Math.random() * 25).toFixed(1)),
        },
    };
};

export const processVideoForDeepfake = async (req, res) => {
    // Logging is helpful, keeping it here.
    console.log("Received request to analyze video.");
    console.log("Request Body:", req.body);
    
    try {
        const { videoId } = req.params;
        const { model } = req.body;

        const validModels = ["SIGLIPV1", "RPPG", "COLORCUES"];
        if (!model || !validModels.includes(model)) {
            return res.status(400).json({
                success: false,
                message: "A valid analysis model is required.",
            });
        }

        const video = await prisma.video.findUnique({ where: { id: videoId } });
        if (!video) {
            return res.status(404).json({ success: false, message: "Video not found" });
        }
        if (video.userId !== req.user.id && req.user.role !== "ADMIN") {
            return res.status(403).json({ success: false, message: "Access denied" });
        }

        // --- START OF CORRECTION ---
        // Use the custom name of the unique constraint from the schema
        const existingAnalysis = await prisma.deepfakeAnalysis.findUnique({
            where: {
                videoId_model_unique_constraint: { // CORRECTED KEY
                    videoId: videoId,
                    model: model,
                },
            },
        });
        // --- END OF CORRECTION ---

        if (existingAnalysis) {
            return res.status(400).json({
                success: false,
                message: `This video has already been analyzed with the ${model} model.`,
                data: { analysis: existingAnalysis },
            });
        }

        await prisma.video.update({
            where: { id: videoId },
            data: { status: "PROCESSING" },
        });

        try {
            const processingDelay = 1000 + Math.random() * 2000;
            console.log(`🤖 Mock Analysis: Processing video ${video.filename} (${videoId}) with model ${model}`);
            await new Promise((resolve) => setTimeout(resolve, processingDelay));
            
            const mockResults = generateMockAnalysis(video.filename, video.size, model);
            
            const analysis = await prisma.deepfakeAnalysis.create({
                data: {
                    videoId: videoId,
                    prediction: mockResults.prediction,
                    confidence: mockResults.confidence,
                    processingTime: mockResults.processing_time,
                    model: model,
                    status: "COMPLETED",
                },
            });

            const updatedVideo = await prisma.video.update({
                where: { id: videoId },
                data: { status: "ANALYZED" },
                include: {
                    analyses: true,
                    user: { select: { id: true, firstName: true, lastName: true, email: true } },
                },
            });

            console.log(`✅ Mock Analysis Complete: Video ${videoId} analyzed successfully`);

            res.json({
                success: true,
                message: "Video analysis completed successfully",
                data: { video: updatedVideo, analysis },
            });
        } catch (analysisError) {
            console.error("🚨 Mock Analysis Error:", analysisError);
            await prisma.video.update({
                where: { id: videoId },
                data: { status: "FAILED" },
            });
            res.status(500).json({ success: false, message: "Analysis failed during processing" });
        }
    } catch (error) {
        console.error("💥 Deepfake processing error:", error);
        // Provide more detailed error in development
        const errorMessage = process.env.NODE_ENV === 'development' ? error.stack : "Internal server error";
        res.status(500).json({ success: false, message: errorMessage });
    }
};

// This function now needs to get ALL analyses for a video.
// Note: This logic is largely handled by the main /api/video/:id route now.
// We can leave this here, but the primary data fetching will use the route below.
export const getAnalysisResults = async (req, res) => {
    try {
        const { videoId } = req.params;

        // Fetching the video will now include ALL its analyses due to schema changes.
        const video = await prisma.video.findUnique({
            where: { id: videoId },
            include: { analyses: true, user: true },
        });

        if (!video) {
            return res.status(404).json({ success: false, message: "Video not found" });
        }

        if (video.userId !== req.user.id && req.user.role !== "ADMIN") {
            return res.status(403).json({ success: false, message: "Access denied" });
        }

        res.json({ success: true, data: { video } });
    } catch (error) {
        console.error("Get analysis error:", error);
        res.status(500).json({ success: false, message: "Internal server error" });
    }
};

// This controller returns VIDEOS, not just analyses. The name is okay.
// It will now correctly include the list of analyses for each video.
export const getAllAnalyses = async (req, res) => {
    try {
        const page = parseInt(req.query.page) || 1;
        const limit = parseInt(req.query.limit) || 10;
        const skip = (page - 1) * limit;

        const whereClause = req.user.role === "ADMIN" ? {} : { userId: req.user.id };

        const [videos, totalCount] = await Promise.all([
            prisma.video.findMany({
                where: whereClause,
                include: {
                    analyses: true, // CHANGED: from analysis to analyses
                    user: { select: { id: true, firstName: true, lastName: true, email: true } },
                },
                orderBy: { createdAt: "desc" },
                skip,
                take: limit,
            }),
            prisma.video.count({ where: whereClause }),
        ]);

        res.json({
            success: true,
            data: {
                videos,
                pagination: { page, limit, totalCount, totalPages: Math.ceil(totalCount / limit) },
            },
        });
    } catch (error) {
        console.error("Get all analyses error:", error);
        res.status(500).json({ success: false, message: "Internal server error" });
    }
};