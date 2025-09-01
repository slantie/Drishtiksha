// src/api/auth/auth.validation.js

import { z } from 'zod';
import { ApiError } from '../../utils/ApiError.js';

export const validate = (schema) => (req, res, next) => {
    try {
        schema.parse({
            body: req.body,
            query: req.query,
            params: req.params,
        });
        next();
    } catch (err) {
        const validationErrors = err.errors.map(e => e.message);
        next(new ApiError(400, "Validation failed", validationErrors));
    }
};

export const signupSchema = z.object({
    body: z.object({
        email: z.string().email('Invalid email address'),
        password: z.string().min(6, 'Password must be at least 6 characters long'),
        firstName: z.string().min(1, 'First name is required'),
        lastName: z.string().min(1, 'Last name is required'),
    }),
});

export const loginSchema = z.object({
    body: z.object({
        email: z.string().email('A valid email is required'),
        password: z.string().min(1, 'Password is required'),
    }),
});

export const updateProfileSchema = z.object({
    body: z.object({
        firstName: z.string().min(1).optional(),
        lastName: z.string().min(1).optional(),
    }),
});

export const updatePasswordSchema = z.object({
    body: z.object({
        currentPassword: z.string().min(1, 'Current password is required'),
        newPassword: z.string().min(6, 'New password must be at least 6 characters'),
    }),
});