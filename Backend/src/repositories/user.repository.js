// src/repositories/user.repository.js

import prisma from "../config/database.js";

const defaultUserSelect = {
    id: true,
    email: true,
    firstName: true,
    lastName: true,
    bio: true,
    phone: true,
    avatar: true,
    role: true,
    isActive: true,
    createdAt: true,
    updatedAt: true,
};

export const userRepository = {
    async findByEmail(email) {
        return await prisma.user.findUnique({ where: { email } });
    },

    async findByEmailWithPassword(email) {
        return await prisma.user.findUnique({ where: { email } });
    },

    async findById(userId) {
        return await prisma.user.findUnique({
            where: { id: userId },
            select: defaultUserSelect,
        });
    },

    async findByIdWithPassword(userId) {
        return await prisma.user.findUnique({ where: { id: userId } });
    },

    async create(userData) {
        const user = await prisma.user.create({ data: userData });
        // eslint-disable-next-line no-unused-vars
        const { password, ...userWithoutPassword } = user;
        return userWithoutPassword;
    },

    async update(userId, updateData) {
        return await prisma.user.update({
            where: { id: userId },
            data: updateData,
            select: defaultUserSelect,
        });
    },
};
