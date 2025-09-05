// src/lib/queryClient.js

import { QueryClient } from "@tanstack/react-query";

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Data is considered fresh for 5 minutes.
      staleTime: 1000 * 60 * 5,
      // Inactive queries are garbage collected after 30 minutes.
      gcTime: 1000 * 60 * 30,
      // Refetch data automatically if the user's network reconnects.
      refetchOnReconnect: "always",
      // Disable automatic refetching when the browser window is focused.
      refetchOnWindowFocus: false,
      // Retry failed requests up to 2 times for server/network errors.
      retry: (failureCount, error) => {
        // Do not retry on 4xx client errors (e.g., 401, 404)
        if (error?.response?.status >= 400 && error?.response?.status < 500) {
          return false;
        }
        // Retry up to 2 times for other errors (e.g., 5xx, network errors)
        return failureCount < 2;
      },
    },
    mutations: {
      // Do not retry mutations by default, as they can have side effects.
      // Retries should be handled on a case-by-case basis.
      retry: 0,
    },
  },
});

export default queryClient;
