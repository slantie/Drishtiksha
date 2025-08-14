// src/lib/queryKeys.js

/**
 * Query key factory for consistent caching across the application
 * This helps with cache management and provides type safety
 */

export const queryKeys = {
    // Auth related queries
    auth: {
        all: ["auth"],
        profile: () => [...queryKeys.auth.all, "profile"],
    },

    // Video related queries
    videos: {
        all: ["videos"],
        lists: () => [...queryKeys.videos.all, "list"],
        list: (filters) => [...queryKeys.videos.lists(), filters],
        details: () => [...queryKeys.videos.all, "detail"],
        detail: (id) => [...queryKeys.videos.details(), id],
        stats: () => [...queryKeys.videos.all, "stats"],
    },

    // User related queries
    users: {
        all: ["users"],
        profile: (id) => [...queryKeys.users.all, "profile", id],
    },
};
