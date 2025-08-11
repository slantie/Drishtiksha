import { prisma } from "../config/database.js";
import path from "path";
import fs from "fs";

// Mock analysis function that generates realistic dummy results
const generateMockAnalysis = (filename, fileSize) => {
    // Generate realistic confidence scores
    const isLikelyReal = Math.random() > 0.3; // 70% chance of being real
    const baseConfidence = isLikelyReal
        ? 0.75 + Math.random() * 0.2
        : 0.6 + Math.random() * 0.3;

    // Add some noise to make it more realistic
    const confidence = Math.min(
        0.98,
        Math.max(0.52, baseConfidence + (Math.random() - 0.5) * 0.1)
    );

    // Processing time based on file size (more realistic)
    const baseProcTime = 2.5 + (fileSize / (1024 * 1024)) * 0.1; // Base 2.5s + 0.1s per MB
    const processingTime = baseProcTime + (Math.random() - 0.5) * 1.0; // Add some variance

    // Determine prediction based on confidence and some logic
    const prediction = confidence > 0.75 ? "REAL" : "FAKE";

    return {
        prediction,
        confidence: parseFloat(confidence.toFixed(4)),
        processing_time: parseFloat(Math.max(1.2, processingTime).toFixed(2)),
        model_version: "siglip-lstm-v1-mock",
        analysis_metadata: {
            frames_analyzed: Math.floor(20 + Math.random() * 40),
            video_duration: parseFloat((5 + Math.random() * 25).toFixed(1)),
            resolution: getRandomResolution(),
            format_detected: getVideoFormat(filename),
        },
    };
};

const getRandomResolution = () => {
    const resolutions = [
        "1920x1080",
        "1280x720",
        "640x480",
        "1024x768",
        "854x480",
    ];
    return resolutions[Math.floor(Math.random() * resolutions.length)];
};

const getVideoFormat = (filename) => {
    const ext = path.extname(filename).toLowerCase();
    const formats = {
        ".mp4": "H.264/MP4",
        ".avi": "AVI Container",
        ".mov": "QuickTime MOV",
        ".wmv": "Windows Media Video",
        ".mkv": "Matroska Video",
    };
    return formats[ext] || "Unknown Format";
};

export const processVideoForDeepfake = async (req, res) => {
    try {
        const { videoId } = req.params;

        // Get video record
        const video = await prisma.video.findUnique({
            where: { id: videoId },
            include: { user: true },
        });

        if (!video) {
            return res.status(404).json({
                success: false,
                message: "Video not found",
            });
        }

        // Check if user owns the video or is admin
        if (video.userId !== req.user.id && req.user.role !== "ADMIN") {
            return res.status(403).json({
                success: false,
                message: "Access denied",
            });
        }

        // Check if analysis already exists
        const existingAnalysis = await prisma.deepfakeAnalysis.findUnique({
            where: { videoId: videoId },
        });

        if (existingAnalysis) {
            return res.status(400).json({
                success: false,
                message: "Video has already been analyzed",
                data: { analysis: existingAnalysis },
            });
        }

        // Update video status to processing
        await prisma.video.update({
            where: { id: videoId },
            data: { status: "PROCESSING" },
        });

        try {
            // Simulate processing delay (1-3 seconds)
            const processingDelay = 1000 + Math.random() * 2000;

            console.log(
                `🤖 Mock Analysis: Processing video ${video.filename} (${videoId})`
            );
            console.log(
                `⏱️  Simulated processing delay: ${(
                    processingDelay / 1000
                ).toFixed(1)}s`
            );

            await new Promise((resolve) =>
                setTimeout(resolve, processingDelay)
            );

            // Verify video file exists (optional check)
            const videoPath = path.resolve(video.filepath);
            if (!fs.existsSync(videoPath)) {
                console.warn(
                    `⚠️  Video file not found: ${videoPath}, continuing with mock analysis...`
                );
            }

            // Generate mock analysis results
            const mockResults = generateMockAnalysis(
                video.filename,
                video.size
            );

            console.log(`📊 Mock Results Generated:`, {
                prediction: mockResults.prediction,
                confidence: `${(mockResults.confidence * 100).toFixed(1)}%`,
                processingTime: `${mockResults.processing_time}s`,
            });

            // Store results in database
            const analysis = await prisma.deepfakeAnalysis.create({
                data: {
                    videoId: videoId,
                    prediction: mockResults.prediction,
                    confidence: mockResults.confidence,
                    processingTime: mockResults.processing_time,
                    modelVersion: mockResults.model_version,
                    status: "COMPLETED",
                },
            });

            // Update video status
            const updatedVideo = await prisma.video.update({
                where: { id: videoId },
                data: {
                    status: "ANALYZED",
                    analysisId: analysis.id,
                },
                include: {
                    analysis: true,
                    user: {
                        select: {
                            id: true,
                            firstName: true,
                            lastName: true,
                            email: true,
                        },
                    },
                },
            });

            console.log(
                `✅ Mock Analysis Complete: Video ${videoId} analyzed successfully`
            );

            res.json({
                success: true,
                message: "Video analysis completed successfully",
                data: {
                    video: updatedVideo,
                    analysis,
                    metadata: mockResults.analysis_metadata,
                },
            });
        } catch (analysisError) {
            console.error("🚨 Mock Analysis Error:", analysisError);

            // Update video status to failed
            await prisma.video.update({
                where: { id: videoId },
                data: { status: "FAILED" },
            });

            res.status(500).json({
                success: false,
                message: "Analysis failed during processing",
                error: analysisError.message,
            });
        }
    } catch (error) {
        console.error("💥 Deepfake processing error:", error);
        res.status(500).json({
            success: false,
            message: "Internal server error",
            error:
                process.env.NODE_ENV === "development"
                    ? error.message
                    : undefined,
        });
    }
};

export const getAnalysisResults = async (req, res) => {
    try {
        const { videoId } = req.params;

        const video = await prisma.video.findUnique({
            where: { id: videoId },
            include: {
                analysis: true,
                user: true,
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

        res.json({
            success: true,
            data: {
                video,
                analysis: video.analysis,
            },
        });
    } catch (error) {
        console.error("Get analysis error:", error);
        res.status(500).json({
            success: false,
            message: "Internal server error",
        });
    }
};

export const getAllAnalyses = async (req, res) => {
    try {
        const page = parseInt(req.query.page) || 1;
        const limit = parseInt(req.query.limit) || 10;
        const skip = (page - 1) * limit;

        const whereClause =
            req.user.role === "ADMIN" ? {} : { userId: req.user.id };

        const [videos, totalCount] = await Promise.all([
            prisma.video.findMany({
                where: whereClause,
                include: {
                    analysis: true,
                    user: {
                        select: {
                            id: true,
                            firstName: true,
                            lastName: true,
                            email: true,
                        },
                    },
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
                pagination: {
                    page,
                    limit,
                    totalCount,
                    totalPages: Math.ceil(totalCount / limit),
                },
            },
        });
    } catch (error) {
        console.error("Get all analyses error:", error);
        res.status(500).json({
            success: false,
            message: "Internal server error",
        });
    }
};
