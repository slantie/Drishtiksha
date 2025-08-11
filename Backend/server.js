import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import authRoutes from "./src/routes/authRoutes.js";
import healthRoutes from "./src/routes/healthRoutes.js";
import videoRoutes from "./src/routes/videoRoutes.js";
import { connectDatabase, disconnectDatabase } from "./src/config/database.js";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 4000;

// CORS configuration for frontend
app.use(
  cors({
    origin: process.env.FRONTEND_URL || process.env.FRONTEND_URL_2,
    credentials: true,
  })
);

app.use(express.json());

// Welcome route
app.get("/", (req, res) => {
  res.json({
    success: true,
    message: "Backend is working!",
    timestamp: new Date().toISOString(),
    version: "1.0.0",
  });
});

// Routes
app.use("/api/auth", authRoutes);
app.use("/api/video", videoRoutes);
app.use("/api/health", healthRoutes);

// Global error handler
app.use((error, req, res, next) => {
  console.error("Global error:", error);
  res.status(error.status || 500).json({
    success: false,
    message:
      process.env.NODE_ENV === "development"
        ? error.message
        : "Internal server error",
  });
});

const startServer = async () => {
  try {
    await connectDatabase();

    app.listen(PORT, () => {
      console.log(`🚀 Server running on http://localhost:${PORT}`);
      console.log(`🗄️ Database connected successfully`);
      console.log(`🌍 Environment: ${process.env.NODE_ENV || "development"}`);
      console.log(`📋 API Routes available:`);
      console.log(`\n🔓 Public Routes:`);
      console.log(`   GET  http://localhost:${PORT}/`);
      console.log(`   GET  http://localhost:${PORT}/api/health`);
      console.log(`   POST http://localhost:${PORT}/api/auth/signup`);
      console.log(`   POST http://localhost:${PORT}/api/auth/login`);
      console.log(`   POST http://localhost:${PORT}/api/auth/logout`);
      console.log(`\n🔒 Protected Routes (require Bearer token):`);
      console.log(`   GET  http://localhost:${PORT}/api/auth/profile`);
      console.log(`   GET  http://localhost:${PORT}/api/auth/profile/full`);
      console.log(`   PUT  http://localhost:${PORT}/api/auth/profile`);
      console.log(`   PUT  http://localhost:${PORT}/api/auth/profile/password`);
      console.log(`   PUT  http://localhost:${PORT}/api/auth/profile/avatar`);
      console.log(`   DEL  http://localhost:${PORT}/api/auth/profile/avatar`);
      console.log(
        `\n💡 Use Authorization: Bearer <token> for protected routes`
      );
    });
  } catch (error) {
    console.error("❌ Failed to start server:", error);
    process.exit(1);
  }
};

// Graceful shutdown
process.on("SIGTERM", async () => {
  console.log("SIGTERM received, shutting down gracefully");
  await disconnectDatabase();
  process.exit(0);
});

process.on("SIGINT", async () => {
  console.log("SIGINT received, shutting down gracefully");
  await disconnectDatabase();
  process.exit(0);
});

startServer();
