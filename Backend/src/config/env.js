// src/config/env.js

import dotenv from 'dotenv';
import { z } from 'zod';

// This is the absolute first step: load the .env file.
dotenv.config();

// Define the schema for all expected environment variables.
const envSchema = z.object({
    NODE_ENV: z.enum(['development', 'production', 'test']).default('development'),
    PORT: z.coerce.number().default(3000),
    DATABASE_URL: z.string().url('DATABASE_URL must be a valid URL'),
    REDIS_URL: z.string().url('REDIS_URL must be a valid URL'),
    MEDIA_PROCESSING_QUEUE_NAME: z.string().default('media-processing-queue'),
    MEDIA_PROGRESS_CHANNEL_NAME: z.string().default('media-progress-events'),
    JWT_SECRET: z.string().min(1, 'JWT_SECRET is required'),
    JWT_EXPIRES_IN: z.string().default('1d'),
    FRONTEND_URL: z.string().url('FRONTEND_URL must be a valid URL'),
    STORAGE_PROVIDER: z.enum(['local', 'cloudinary']).default('local'),
    BASE_URL: z.string().url('BASE_URL must be a valid URL'),
    LOCAL_STORAGE_PATH: z.string().default('public/media'),
    SERVER_URL: z.string().url('SERVER_URL must be a valid URL for the ML server'),
    SERVER_API_KEY: z.string().min(1, 'SERVER_API_KEY is required'),
});

const validatedEnv = envSchema.safeParse(process.env);

if (!validatedEnv.success) {
    console.error('❌ FATAL: Invalid environment variables:');
    // Use console.error because the logger may not be initialized yet.
    console.error(validatedEnv.error.flatten().fieldErrors);
    process.exit(1); // Fail fast
}

// Export the single, validated config object. This file has no other dependencies.
export const config = validatedEnv.data;