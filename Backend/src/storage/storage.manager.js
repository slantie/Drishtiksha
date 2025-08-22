// src/storage/storage.manager.js

import dotenv from "dotenv";
import logger from "../utils/logger.js";

dotenv.config({ path: "./.env" });

const providerType = process.env.STORAGE_PROVIDER || "cloudinary";

let storageManager;

if (providerType === "local") {
    const { default: localProvider } = await import("./local.provider.js");
    storageManager = localProvider;
    logger.info("📦 Storage Manager: Using 'local' filesystem provider.");
} else {
    const { default: cloudinaryProvider } = await import("./cloudinary.provider.js");
    storageManager = cloudinaryProvider;
    logger.info("☁️ Storage Manager: Using 'cloudinary' provider.");
}

export default storageManager;