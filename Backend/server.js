// Backend/server.js

import dotenv from "dotenv";
import { app } from "./src/app.js";
import { connectDatabase, disconnectDatabase } from "./src/config/database.js";

// Load environment variables from .env file
dotenv.config({
    path: "./.env",
});

const PORT = process.env.PORT || 4000;

const startServer = async () => {
    try {
        // 1. Connect to the database
        await connectDatabase();
        console.log("🗄️  Database connected successfully.");

        // 2. Start the Express server
        const server = app.listen(PORT, () => {
            console.log(`\n🚀 Server is running at: http://localhost:${PORT}`);
            console.log(
                `   Environment: ${process.env.NODE_ENV || "development"}`
            );
        });

        // Graceful shutdown logic
        const shutdown = async (signal) => {
            console.log(`\n${signal} received. Shutting down gracefully...`);
            server.close(async () => {
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
