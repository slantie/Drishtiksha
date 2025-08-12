// src/services/auth.service.js

import { userRepository } from "../repositories/user.repository.js";
import { hashPassword, comparePassword } from "../utils/password.js";
import { generateToken } from "../utils/jwt.js";
import { ApiError } from "../utils/ApiError.js";

export const authService = {
    async registerUser({ email, firstName, lastName, password }) {
        const existingUser = await userRepository.findByEmail(
            email.toLowerCase()
        );
        if (existingUser) {
            throw new ApiError(409, "User with this email already exists");
        }

        const hashedPassword = await hashPassword(password);

        const newUser = await userRepository.create({
            email: email.toLowerCase(),
            firstName: firstName.trim(),
            lastName: lastName.trim(),
            password: hashedPassword,
        });

        const token = generateToken({
            userId: newUser.id,
            email: newUser.email,
            role: newUser.role,
        });

        return { user: newUser, token };
    },

    async loginUser(email, password) {
        const user = await userRepository.findByEmailWithPassword(
            email.toLowerCase()
        );
        if (!user || !user.isActive) {
            throw new ApiError(401, "Invalid email or password");
        }

        const isPasswordValid = await comparePassword(password, user.password);
        if (!isPasswordValid) {
            throw new ApiError(401, "Invalid email or password");
        }

        const token = generateToken({
            userId: user.id,
            email: user.email,
            role: user.role,
        });

        // eslint-disable-next-line no-unused-vars
        const { password: _, ...userWithoutPassword } = user;

        return { user: userWithoutPassword, token };
    },

    async getUserProfile(userId) {
        const user = await userRepository.findById(userId);
        if (!user) {
            throw new ApiError(404, "User not found");
        }
        return user;
    },

    async updateUserProfile(userId, profileData) {
        return await userRepository.update(userId, profileData);
    },

    async changePassword(userId, currentPassword, newPassword) {
        const user = await userRepository.findByIdWithPassword(userId);
        if (!user) {
            throw new ApiError(404, "User not found");
        }

        const isPasswordValid = await comparePassword(
            currentPassword,
            user.password
        );
        if (!isPasswordValid) {
            throw new ApiError(401, "Current password is incorrect");
        }

        const hashedNewPassword = await hashPassword(newPassword);
        await userRepository.update(userId, { password: hashedNewPassword });
    },

    async updateUserAvatar(userId, avatarUrl) {
        return await userRepository.update(userId, { avatar: avatarUrl });
    },

    async removeUserAvatar(userId) {
        return await userRepository.update(userId, { avatar: null });
    },
};
