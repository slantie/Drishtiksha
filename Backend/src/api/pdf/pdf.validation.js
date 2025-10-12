// src/api/pdf/pdf.validation.js

import { z } from 'zod';

/**
 * Validation middleware wrapper
 */
export const validate = (schema) => async (req, res, next) => {
    try {
        await schema.parseAsync({
            body: req.body,
            params: req.params,
            query: req.query,
        });
        next();
    } catch (error) {
        return res.status(400).json({
            success: false,
            message: 'Validation failed',
            errors: error.errors,
        });
    }
};

/**
 * Schema for media ID parameter validation
 */
export const mediaIdParamSchema = z.object({
    params: z.object({
        mediaId: z.string().uuid('Invalid media ID format'),
    }),
});

/**
 * Schema for analysis run ID parameter validation
 */
export const analysisRunIdParamSchema = z.object({
    params: z.object({
        analysisRunId: z.string().uuid('Invalid analysis run ID format'),
    }),
});
