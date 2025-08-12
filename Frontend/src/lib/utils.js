// src/lib/utils.js

import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

// Merge Tailwind and conditional classes
export function cn(...inputs) {
    return twMerge(clsx(inputs));
}
