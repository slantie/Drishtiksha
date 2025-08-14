// src/lib/queryClient.js

import { QueryClient } from "@tanstack/react-query";

export const queryClient = new QueryClient({
    defaultOptions: {
        queries: {
            // Time in milliseconds before data is considered stale
            staleTime: 1000 * 60 * 5, // 5 minutes
            // Time in milliseconds before inactive queries are garbage collected
            gcTime: 1000 * 60 * 30, // 30 minutes (formerly cacheTime)
            // Retry failed requests 3 times with exponential backoff
            retry: (failureCount, error) => {
                // Don't retry on 4xx errors (except 408, 429)
                if (
                    error?.response?.status >= 400 &&
                    error?.response?.status < 500
                ) {
                    if (
                        error.response.status === 408 ||
                        error.response.status === 429
                    ) {
                        return failureCount < 2;
                    }
                    return false;
                }
                // Retry up to 3 times for other errors
                return failureCount < 3;
            },
            // Disable automatic refetching on window focus by default
            refetchOnWindowFocus: false,
            // Enable refetching on reconnect
            refetchOnReconnect: "always",
        },
        mutations: {
            // Retry mutations once on failure
            retry: 1,
        },
    },
});

export default queryClient;
