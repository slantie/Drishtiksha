// Backend/assets.server.js

import express from "express";
import cors from "cors";
import helmet from "helmet";
import morgan from "morgan";
import path from "path";
import { fileURLToPath } from "url";
import { config } from "./src/config/env.js";
import logger from "./src/utils/logger.js";

const app = express();
const PORT = config.ASSETS_PORT;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// --- Core Middleware ---
app.use(helmet({ crossOriginResourcePolicy: { policy: "cross-origin" } }));
app.use(cors({ origin: [config.FRONTEND_URL, "http://localhost:5173"] })); // Allow the frontend to request files
app.use(morgan("dev", { stream: logger.stream }));

// --- Static File Serving Logic ---
// This is the server's ONLY job.
const projectRoot = path.resolve(__dirname); // Absolute path to Backend directory
const staticDirPath = path.join(projectRoot, config.LOCAL_STORAGE_PATH);

// Create a URL path segment from the storage path (e.g., 'public/media' -> '/media')
const urlPath = config.LOCAL_STORAGE_PATH.split("/").slice(1).join("/");

logger.info(
  `Asset server will serve files from URL '/${urlPath}' mapped to directory '${staticDirPath}'`
);

// Mount the static directory
app.use(`/${urlPath}`, express.static(staticDirPath));

// --- Start the Server ---
app.listen(PORT, () => {
  logger.info(`📦 Asset Server is running at: http://localhost:${PORT}`);
});
