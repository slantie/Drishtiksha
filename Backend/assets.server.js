// assets.server.js

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
app.use(cors({ origin: config.FRONTEND_URL })); // Allow the frontend to request files
app.use(morgan("dev", { stream: logger.stream }));

// --- Static File Serving Logic ---
// This is the server's ONLY job.
const projectRoot = __dirname;
const staticDirPath = path.join(projectRoot, config.LOCAL_STORAGE_PATH);
const staticUrlPath = config.LOCAL_STORAGE_PATH.replace(/^public\//, "");

// Mount the static directory
app.use(`/${staticUrlPath}`, express.static(staticDirPath));

logger.info(
  `Asset server will serve files from URL '/${staticUrlPath}' mapped to directory '${staticDirPath}'`
);

// --- Start the Server ---
app.listen(PORT, () => {
  logger.info(`📦 Asset Server is running at: http://localhost:${PORT}`);
});
