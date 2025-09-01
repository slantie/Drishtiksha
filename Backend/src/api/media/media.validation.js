// src/api/media/media.validation.js

import { z } from 'zod';
import { ApiError } from '../../utils/ApiError.js';

export const validate = (schema) => (req, res, next) => {
    try {
        schema.parse({ body: req.body, params: req.params });
        next();
    } catch (err) {
        const validationErrors = err.errors.map(e => e.message);
        next(new ApiError(400, "Validation failed", validationErrors));
    }
};

export const mediaUpdateSchema = z.object({
    body: z.object({
        description: z.string().max(500, 'Description cannot exceed 500 characters.').optional(),
    }),
    params: z.object({
        id: z.string().uuid('Invalid media ID format.'),
    }),
});

export const mediaIdParamSchema = z.object({
    params: z.object({
        id: z.string().uuid('Invalid media ID format.'),
    }),
});