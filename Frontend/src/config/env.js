// src/config/env.js

import { z } from "zod";

// Define the schema for all expected public environment variables.
// In Vite, these MUST be prefixed with `VITE_`.
const envSchema = z.object({
  // The public-facing name of the project, used in titles and headers.
  VITE_PROJECT_NAME: z.string().min(1).default("Drishtiksha"),

  // A short description of the project, used in HTML meta tags.
  VITE_PROJECT_DESC: z.string().min(1).default("Deepfake Detection"),

  // The full URL to the backend API service. This is the most critical variable.
  VITE_BACKEND_URL: z.string().url("A valid VITE_BACKEND_URL is required."),

  // The backend URL version to the backend API service. This is the most critical variable.
  VITE_BACKEND_URL_VERSION: z.string(),
  // .url("A valid VITE_BACKEND_URL_VERSION is required."),
});

// Vite exposes environment variables on `import.meta.env`.
const validatedEnv = envSchema.safeParse(import.meta.env);

if (!validatedEnv.success) {
  const errorMessages = Object.entries(validatedEnv.error.flatten().fieldErrors)
    .map(([key, value]) => `- ${key}: ${value.join(", ")}`)
    .join("\n");

  // Throw a clear, developer-friendly error that stops the application from starting.
  throw new Error(
    `❌ Invalid environment variables found in .env file:\n${errorMessages}`
  );
}

// Export the single, validated, and type-safe config object.
export const config = validatedEnv.data;
