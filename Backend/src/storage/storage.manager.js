// src/storage/storage.manager.js

import { config } from "../config/env.js";
import logger from "../utils/logger.js";
import localProvider from "./local.provider.js";
import cloudinaryProvider from "./cloudinary.provider.js";

let storageManager;

// This logic now correctly selects the storage provider based on the .env configuration.
if (config.STORAGE_PROVIDER === "local") {
  storageManager = localProvider;
  logger.info(
    "📦 Storage Manager initialized with 'local' filesystem provider."
  );
} else {
  storageManager = cloudinaryProvider;
  logger.info("☁️ Storage Manager initialized with 'cloudinary' provider.");
}

export default storageManager;
