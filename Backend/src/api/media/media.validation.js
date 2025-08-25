// src/api/media/media.validation.js

import { z } from "zod";

const validate = (schema) => (req, res, next) => {
    try {
        schema.parse({ body: req.body, params: req.params });
        next();
    } catch (err) {
        return res.status(400).json({ success: false, errors: err.errors });
    }
};

// RENAMED: from videoUpdateSchema to mediaUpdateSchema
const mediaUpdateSchema = z.object({
    body: z.object({
        description: z.string().optional(),
        filename: z.string().min(3).optional(),
    }),
    params: z.object({
        // UPDATED: Changed error message to be generic
        id: z.string().uuid("Invalid media ID format"),
    }),
});

// UPDATED: This schema is now more flexible. The specific model validation
// will be handled in the service layer, as it depends on the media type.
const analysisRequestSchema = z.object({
    body: z.object({
        type: z.enum(
            ["QUICK", "DETAILED", "FRAMES", "VISUALIZE", "COMPREHENSIVE"],
            {
                required_error: "Analysis type is required",
                invalid_type_error: "Invalid analysis type",
            }
        ),
        // REMOVED: The hardcoded model enum is gone. Now accepts any string.
        model: z.string().optional(),
    }),
    params: z.object({
        id: z.string().uuid("Invalid media ID format"),
    }),
});

export { validate, mediaUpdateSchema, analysisRequestSchema };
