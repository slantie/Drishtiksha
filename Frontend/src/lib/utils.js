/**
 * @file src/lib/utils.js
 * @description Utility for merging Tailwind and conditional classes
 */

"use client";

import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

// Merge Tailwind and conditional classes
export function cn(...inputs) {
    return twMerge(clsx(inputs));
}