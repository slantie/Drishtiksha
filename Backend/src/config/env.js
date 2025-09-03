// src/config/env.js

import dotenv from "dotenv";
import { z } from "zod";

dotenv.config();

const envSchema = z
  .object({
    NODE_ENV: z
      .enum(["development", "production", "test"])
      .default("development"),
    PORT: z.coerce.number().default(3000),
    API_BASE_URL: z
      .string()
      .url({ message: "API_BASE_URL must be a valid URL" }),

    // ASSET SERVER (for local storage)
    ASSETS_PORT: z.coerce.number().default(3001),
    ASSETS_BASE_URL: z
      .string()
      .url({ message: "ASSETS_BASE_URL must be a valid URL" }),

    // CORE SERVICES
    DATABASE_URL: z
      .string()
      .url({ message: "DATABASE_URL must be a valid URL" }),
    REDIS_URL: z.string().url({ message: "REDIS_URL must be a valid URL" }),

    // QUEUE
    MEDIA_PROCESSING_QUEUE_NAME: z.string().default("media-processing-queue"),
    MEDIA_PROGRESS_CHANNEL_NAME: z.string().default("media-progress-events"),

    // SECURITY
    JWT_SECRET: z.string().min(1, "JWT_SECRET is required"),
    JWT_EXPIRES_IN: z.string().default("1d"),
    BCRYPT_ROUNDS: z.coerce.number().int().positive().default(12),

    // CORS
    FRONTEND_URL: z
      .string()
      .url({ message: "FRONTEND_URL must be a valid URL" }),

    // FILE STORAGE STRATEGY
    STORAGE_PROVIDER: z.enum(["local", "cloudinary"]).default("local"),
    LOCAL_STORAGE_PATH: z.string().default("public/media"),

    // ML SERVER
    SERVER_URL: z
      .string()
      .url({ message: "SERVER_URL must be a valid URL for the ML server" }),
    SERVER_API_KEY: z.string().min(1, "SERVER_API_KEY is required"),

    // CLOUDINARY (Conditionally required)
    CLOUDINARY_CLOUD_NAME: z.string().optional(),
    CLOUDINARY_API_KEY: z.string().optional(),
    CLOUDINARY_API_SECRET: z.string().optional(),
  })
  // This is the critical hardening step.
  // It ensures that if Cloudinary is selected, its keys MUST be provided.
  .superRefine((data, ctx) => {
    if (data.STORAGE_PROVIDER === "cloudinary") {
      if (!data.CLOUDINARY_CLOUD_NAME) {
        ctx.addIssue({
          code: z.custom,
          path: ["CLOUDINARY_CLOUD_NAME"],
          message:
            "CLOUDINARY_CLOUD_NAME is required when STORAGE_PROVIDER is 'cloudinary'",
        });
      }
      if (!data.CLOUDINARY_API_KEY) {
        ctx.addIssue({
          code: z.custom,
          path: ["CLOUDINARY_API_KEY"],
          message:
            "CLOUDINARY_API_KEY is required when STORAGE_PROVIDER is 'cloudinary'",
        });
      }
      if (!data.CLOUDINARY_API_SECRET) {
        ctx.addIssue({
          code: z.custom,
          path: ["CLOUDINARY_API_SECRET"],
          message:
            "CLOUDINARY_API_SECRET is required when STORAGE_PROVIDER is 'cloudinary'",
        });
      }
    }
  });

const validatedEnv = envSchema.safeParse(process.env);

if (!validatedEnv.success) {
  console.error("❌ FATAL: Invalid environment variables:");
  // This provides a clean, readable list of all configuration errors.
  console.error(validatedEnv.error.flatten().fieldErrors);
  process.exit(1);
}

export const config = validatedEnv.data;
