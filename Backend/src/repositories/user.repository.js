// src/repositories/user.repository.js

import { prisma } from '../config/index.js';

const defaultUserSelect = {
    id: true,
    email: true,
    firstName: true,
    lastName: true,
    role: true,
    isActive: true,
    createdAt: true,
    updatedAt: true,
};

export const userRepository = {
    async findByEmailWithPassword(email) {
        return prisma.user.findUnique({ where: { email } });
    },

    async findById(userId) {
        return prisma.user.findUnique({
            where: { id: userId },
            select: defaultUserSelect,
        });
    },
    
    async findByIdWithPassword(userId) {
        return prisma.user.findUnique({ where: { id: userId } });
    },

    async create(userData) {
        return prisma.user.create({
            data: userData,
            select: defaultUserSelect,
        });
    },

    async update(userId, updateData) {
        return prisma.user.update({
            where: { id: userId },
            data: updateData,
            select: defaultUserSelect,
        });
    },
};