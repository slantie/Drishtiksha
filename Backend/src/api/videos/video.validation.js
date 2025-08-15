// src/api/videos/video.validation.js

import { z } from "zod";

const validate = (schema) => (req, res, next) => {
    try {
        schema.parse({ body: req.body, params: req.params });
        next();
    } catch (err) {
        return res.status(400).json({ success: false, errors: err.errors });
    }
};

const videoUpdateSchema = z.object({
    body: z.object({
        description: z.string().optional(),
        filename: z.string().min(3).optional(),
    }),
    params: z.object({
        id: z.string().uuid("Invalid video ID format"),
    }),
});

const analysisRequestSchema = z.object({
    body: z.object({
        type: z.enum(["QUICK", "DETAILED", "FRAMES", "VISUALIZE"], {
            required_error: "Analysis type is required",
            invalid_type_error: "Invalid analysis type",
        }),
        model: z
            .enum(["SIGLIP_LSTM_V1", "SIGLIP_LSTM_V3", "COLOR_CUES_LSTM_V1"])
            .optional(),
    }),
    params: z.object({
        id: z.string().uuid("Invalid video ID format"),
    }),
});

export { validate, videoUpdateSchema, analysisRequestSchema };
