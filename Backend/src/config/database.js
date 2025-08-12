// src/config/database.js

import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient({
    log: ["error", "warn"],
});

const connectDatabase = async () => {
    try {
        await prisma.$connect();
        console.log("🗄️  Database connected successfully.");
    } catch (error) {
        console.error("❌ Database connection failed:", error);
        process.exit(1);
    }
};

const disconnectDatabase = async () => {
    await prisma.$disconnect();
    console.log("🗄️  Database disconnected.");
};

export default prisma;

export { connectDatabase, disconnectDatabase };
