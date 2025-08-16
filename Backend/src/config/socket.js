// src/config/socket.js

import { Server } from "socket.io";
import { verifyToken } from "../utils/jwt.js";
import logger from "../utils/logger.js";

export const initializeSocketIO = (httpServer) => {
    const io = new Server(httpServer, {
        cors: {
            origin: process.env.FRONTEND_URL || "*",
            methods: ["GET", "POST"],
        },
    });

    // REASON: This middleware protects the socket connection, ensuring only authenticated users can connect.
    io.use((socket, next) => {
        const token = socket.handshake.auth.token;
        if (!token) {
            return next(new Error("Authentication error: No token provided."));
        }
        try {
            const decoded = verifyToken(token);
            socket.user = decoded; // Attach user payload to the socket
            next();
        } catch (err) {
            return next(new Error("Authentication error: Invalid token."));
        }
    });

    io.on("connection", (socket) => {
        logger.info(
            `✅ User connected via WebSocket: ${socket.user.email} (ID: ${socket.id})`
        );

        // REASON: Each user joins a private room based on their user ID. This ensures that a user only
        // receives real-time updates for their own videos, maintaining data privacy.
        socket.join(socket.user.userId);

        socket.on("disconnect", () => {
            logger.info(
                `🔌 User disconnected: ${socket.user.email} (ID: ${socket.id})`
            );
        });
    });

    return io;
};
