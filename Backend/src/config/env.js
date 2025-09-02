// src/config/env.js

import dotenv from "dotenv";
import { z } from "zod";

dotenv.config();

const envSchema = z.object({
  NODE_ENV: z
    .enum(["development", "production", "test"])
    .default("development"),
  PORT: z.coerce.number().default(3000),
  API_BASE_URL: z.string().url({ message: "API_BASE_URL must be a valid URL" }),
  ASSETS_PORT: z.coerce.number().default(3001),
  ASSETS_BASE_URL: z
    .string()
    .url({ message: "ASSETS_BASE_URL must be a valid URL" }),
  DATABASE_URL: z.string().url({ message: "DATABASE_URL must be a valid URL" }),
  REDIS_URL: z.string().url({ message: "REDIS_URL must be a valid URL" }),
  MEDIA_PROCESSING_QUEUE_NAME: z.string().default("media-processing-queue"),
  MEDIA_PROGRESS_CHANNEL_NAME: z.string().default("media-progress-events"),
  JWT_SECRET: z.string().min(1, "JWT_SECRET is required"),
  JWT_EXPIRES_IN: z.string().default("1d"),
  FRONTEND_URL: z.string().url({ message: "FRONTEND_URL must be a valid URL" }),
  STORAGE_PROVIDER: z.enum(["local", "cloudinary"]).default("cloudinary"),
  LOCAL_STORAGE_PATH: z.string().default("public/media"),
  SERVER_URL: z
    .string()
    .url({ message: "SERVER_URL must be a valid URL for the ML server" }),
  SERVER_API_KEY: z.string().min(1, "SERVER_API_KEY is required"),
  CLOUDINARY_CLOUD_NAME: z.string(),
  CLOUDINARY_API_KEY: z.string(),
  CLOUDINARY_API_SECRET: z.string(),
});

const validatedEnv = envSchema.safeParse(process.env);

if (!validatedEnv.success) {
  console.error("❌ FATAL: Invalid environment variables:");
  console.error(validatedEnv.error.flatten().fieldErrors);
  process.exit(1);
}

export const config = validatedEnv.data;
