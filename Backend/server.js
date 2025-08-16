// Backend/server.js

import dotenv from "dotenv";
import { createServer } from "http"; // ADDED
import { app } from "./src/app.js";
import { connectDatabase, disconnectDatabase } from "./src/config/database.js";
import { initializeSocketIO } from "./src/config/socket.js"; // ADDED

dotenv.config({ path: "./.env" });

const PORT = process.env.PORT || 4000;

// ADDED: Create an HTTP server to attach both Express and Socket.IO
const httpServer = createServer(app);

// ADDED: Initialize Socket.IO and pass the server instance
const io = initializeSocketIO(httpServer);
app.set("io", io); // Make io accessible in request handlers

const startServer = async () => {
    try {
        await connectDatabase();

        // CHANGED: Use the httpServer to listen for requests
        httpServer.listen(PORT, () => {
            console.log(`\n🚀 Server is running at: http://localhost:${PORT}`);
            console.log(
                `   Environment: ${process.env.NODE_ENV || "development"}`
            );
        });

        const shutdown = async (signal) => {
            console.log(`\n${signal} received. Shutting down gracefully...`);
            httpServer.close(async () => {
                await disconnectDatabase();
                console.log("🔌 Server and database connections closed.");
                process.exit(0);
            });
        };

        process.on("SIGTERM", () => shutdown("SIGTERM"));
        process.on("SIGINT", () => shutdown("SIGINT"));
    } catch (error) {
        console.error("❌ Failed to start server:", error);
        process.exit(1);
    }
};

startServer();
