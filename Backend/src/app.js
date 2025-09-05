// Backend/src/app.js

import express from "express";
import cors from "cors";
import cookieParser from "cookie-parser";
import helmet from "helmet";
import morgan from "morgan";
import { config } from "./config/env.js";
import logger from "./utils/logger.js";
import { errorMiddleware } from "./middleware/error.middleware.js";
import { apiRateLimiter } from "./middleware/security.middleware.js";
import authRoutes from "./api/auth/auth.routes.js";
import mediaRoutes from "./api/media/media.routes.js";
import monitoringRoutes from "./api/monitoring/monitoring.routes.js";

const app = express();

app.use(helmet());
app.use(cors({ origin: config.FRONTEND_URL, credentials: true }));
app.use("/api", apiRateLimiter);
app.use(morgan("dev", { stream: logger.stream }));
app.use(express.json({ limit: "20kb" }));
app.use(express.urlencoded({ extended: true, limit: "20kb" }));
app.use(cookieParser());

// --- API Routes ---
app.use("/api/v1/auth", authRoutes);
app.use("/api/v1/media", mediaRoutes);
app.use("/api/v1/monitoring", monitoringRoutes);

app.get("/", (req, res) => {
  res.status(200).json({
    success: true,
    message: "Drishtiksha Backend API is alive and running!",
  });
});

app.use(errorMiddleware);

export { app };
