// src/api/auth/auth.validation.js

import { z } from "zod";

const validate = (schema) => (req, res, next) => {
    try {
        schema.parse({
            body: req.body,
            query: req.query,
            params: req.params,
        });
        next();
    } catch (err) {
        return res.status(400).json({ success: false, errors: err.errors });
    }
};

const signupSchema = z.object({
    body: z.object({
        email: z.string().email("Invalid email address"),
        password: z
            .string()
            .min(6, "Password must be at least 6 characters long"),
        firstName: z.string().min(2, "First name is required"),
        lastName: z.string().min(2, "Last name is required"),
    }),
});

const loginSchema = z.object({
    body: z.object({
        email: z.string().email("A valid email is required"),
        password: z.string().min(1, "Password is required"),
    }),
});

const updateProfileSchema = z.object({
    body: z.object({
        firstName: z.string().min(2).optional(),
        lastName: z.string().min(2).optional(),
        bio: z.string().max(500).optional(),
        phone: z
            .string()
            .regex(
                /^\+91\s?[6-9]\d{9}$/,
                "Phone number must be in format +91 XXXXXXXXXX (Indian mobile numbers only)"
            )
            .optional()
            .nullable(),
    }),
});

const updatePasswordSchema = z.object({
    body: z
        .object({
            currentPassword: z.string().min(1),
            newPassword: z.string().min(6),
            confirmPassword: z.string().min(6),
        })
        .refine((data) => data.newPassword === data.confirmPassword, {
            message: "New passwords do not match",
            path: ["confirmPassword"],
        }),
});

const updateAvatarSchema = z.object({
    body: z.object({
        avatar: z.string().url("Must be a valid URL"),
    }),
});

export {
    validate,
    signupSchema,
    loginSchema,
    updateProfileSchema,
    updatePasswordSchema,
    updateAvatarSchema,
};
