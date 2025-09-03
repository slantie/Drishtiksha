// src/config/socket.js

import { Server } from "socket.io";
import { verifyToken } from "../utils/jwt.js";
import logger from "../utils/logger.js";
import { config } from "./env.js";

export const initializeSocketIO = (httpServer) => {
  const io = new Server(httpServer, {
    cors: {
      origin: config.FRONTEND_URL,
      methods: ["GET", "POST"],
      credentials: true,
    },
  });

  io.use((socket, next) => {
    const token = socket.handshake.auth.token;
    if (!token) {
      return next(new Error("Authentication error: No token provided."));
    }
    try {
      const decoded = verifyToken(token);
      socket.user = decoded;
      next();
    } catch (err) {
      return next(new Error("Authentication error: Invalid token."));
    }
  });

  io.on("connection", (socket) => {
    if (!socket.user?.userId) {
      logger.warn("[SocketIO] Connection with invalid user payload in socket.");
      socket.disconnect();
      return;
    }
    logger.info(
      `[SocketIO] ✅ User connected: ${socket.user.email} (Socket ID: ${socket.id})`
    );
    socket.join(socket.user.userId);
    socket.on("disconnect", () => {
      logger.info(
        `[SocketIO] 🔌 User disconnected: ${socket.user.email} (Socket ID: ${socket.id})`
      );
    });
  });

  return io;
};
