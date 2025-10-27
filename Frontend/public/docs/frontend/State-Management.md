# Frontend: State Management Architecture

## 1. Overview

In a modern web application, "state" refers to all the data that can change over time and affects what the user sees. We categorize state into two main types:

1.  **Server State:** Data that lives on our backend server (e.g., the user's profile, the list of uploaded videos, analysis results). This is the "single source of truth."
2.  **UI State:** Data that is temporary and specific to the user's current interaction with the interface (e.g., "is this modal open?", "what text is currently in the search bar?").

Our state management strategy is built on a powerful principle: **treat the backend as the true state and the frontend as a synchronized cache.** This makes the application faster, more resilient, and simpler to reason about.

- For **Server State**, we use **TanStack React Query**.
- For global **UI State** (like user authentication), we use **React Context**.

---

## 2. Technical Deep Dive

### 2.1. The Duality of State

Separating state into these two categories is the most important architectural decision in our frontend.

- **Server State:**
  - **Characteristics:** Asynchronous, fetched over the network, can be modified by other users, can become "stale."
  - **Managed by:** **TanStack React Query**.
- **UI State:**
  - **Characteristics:** Synchronous, exists only in the browser, controlled entirely by the user's actions.
  - **Managed by:** **React Hooks** (`useState`, `useReducer`) for local component state, and **React Context** for global state.

### 2.2. React Query: The Server State Manager

React Query is the backbone of our data fetching and state synchronization. It is configured globally in `src/lib/queryClient.js`.

**Core Concepts:**

- **Queries (`useQuery`):** Used for fetching (GET) data. React Query automatically handles caching, background refetching, and stale-data management.
- **Mutations (`useMutation`):** Used for creating, updating, or deleting (POST, PUT, PATCH, DELETE) data. Mutations handle loading/error states and provide hooks to intelligently update the query cache after a successful operation.
- **Query Keys (`queryKeys.js`):** A system of unique, structured keys used to identify and manage cached data.

**Example: Fetching a User's Media (`useMediaQuery.jsx`)**

```javascript
// src/hooks/useMediaQuery.jsx
import { useQuery } from "@tanstack/react-query";
import { mediaApi } from "../services/api/media.api.js";
import { queryKeys } from "../lib/queryKeys.js";

export const useMediaQuery = () => {
  return useQuery({
    queryKey: queryKeys.media.lists(), // ["media", "list"]
    queryFn: mediaApi.getAll, // The API call function
    select: (response) => response.data, // Extract the data from the API response
  });
};
```

**How it works in a component:**

```jsx
// src/pages/Dashboard.jsx
import { useMediaQuery } from "../hooks/useMediaQuery.jsx";

function Dashboard() {
  const { data: mediaItems, isLoading, error } = useMediaQuery();

  if (isLoading) return <DashboardSkeleton />;
  if (error) return <ErrorMessage message={error.message} />;

  return <DataTable data={mediaItems} ... />;
}
```

**Benefits of this approach:**

- **Declarative Data Fetching:** The component simply declares what data it needs. React Query handles _how_ and _when_ to fetch it.
- **Automatic Caching:** If the user navigates away and comes back, the data is served instantly from the cache while a fresh copy is fetched in the background.
- **Simplified State:** No more manual `useState` for loading, error, and data states. `useQuery` provides all three.

#### Smart Polling with `refetchInterval`

For pages that need to display live progress (like the `Results.jsx` page), we use React Query's `refetchInterval` option to implement **smart polling**.

```javascript
// src/hooks/useMediaQuery.jsx
export const useMediaItemQuery = (mediaId) => {
  const { isConnected: isSocketConnected } = useSocketStatus();

  return useQuery({
    queryKey: queryKeys.media.detail(mediaId),
    queryFn: () => mediaApi.getById(mediaId),
    refetchInterval: (query) => {
      const media = query.state.data;
      const status = media?.status;

      // 1. Stop polling if analysis is complete.
      if (!["QUEUED", "PROCESSING"].includes(status)) {
        return false;
      }

      // 2. Poll slowly if WebSocket is connected (as a fallback).
      if (isSocketConnected) {
        return 30000; // 30 seconds
      }

      // 3. Poll actively if WebSocket is disconnected.
      return 5000; // 5 seconds
    },
  });
};
```

This logic is highly efficient: it polls only when necessary and adapts its frequency based on the real-time connection status.

#### Mutations and Cache Invalidation

When we change data on the server (e.g., deleting a media item), we must tell React Query to update its cache. This is done with `queryClient.invalidateQueries`.

**Example: Deleting Media (`useMediaQuery.jsx`)**

```javascript
// src/hooks/useMediaQuery.jsx
export const useDeleteMediaMutation = () => {
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  return useMutation({
    mutationFn: mediaApi.delete,
    onSuccess: (_, mediaId) => {
      showToast.success("Media deleted successfully.");
      // 1. Invalidate the list of all media to trigger a refetch on the dashboard.
      queryClient.invalidateQueries({ queryKey: queryKeys.media.lists() });
      // 2. Immediately remove the detailed view from the cache.
      queryClient.removeQueries({ queryKey: queryKeys.media.detail(mediaId) });
      navigate("/dashboard");
    },
    onError: (error) => {
      showToast.error(error.message || "Failed to delete media.");
    },
  });
};
```

### 2.3. React Context: Global UI State (`AuthContext.jsx`)

For state that needs to be accessible by any component in the application, we use React Context. Our primary use case is for **authentication**.

**`AuthContext.jsx` Responsibilities:**

1.  **State Storage:** Holds the current `user` object, `token`, and `isAuthenticated` flag.
2.  **State Derivation:** Reads the initial token and user from `authStorage` (`localStorage`/`sessionStorage`).
3.  **Action Provider:** Exposes functions like `login`, `logout`, and `signup` to all components.
4.  **Orchestration:** The `login` function is a workflow: it calls the `useLoginMutation`, stores the token on success, updates the socket connection, and navigates the user.

**Simplified `AuthContext` Logic:**

```jsx
// src/contexts/AuthContext.jsx
export const AuthProvider = ({ children }) => {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const [localToken, setLocalToken] = useState(authStorage.get().token);

  // Query for the user's profile, but only if a token exists.
  const {
    data: user,
    isLoading,
    isError,
    isSuccess,
  } = useProfileQuery(localToken);

  // Effect to manage the WebSocket connection based on token state.
  useEffect(() => {
    if (localToken) {
      socketService.connect(localToken);
    } else {
      socketService.disconnect();
    }
  }, [localToken]);

  // The login function orchestrates the entire login flow.
  const login = useCallback(
    async (email, password) => {
      const response = await loginMutation.mutateAsync({ email, password });
      const { token, user } = response.data;

      // 1. Store token
      authStorage.set({ token, user });
      setLocalToken(token);

      // 2. Optimistically update React Query cache
      queryClient.setQueryData(queryKeys.auth.profile(), user);

      // 3. Navigate to dashboard
      navigate("/dashboard");
    },
    [loginMutation, queryClient, navigate]
  );

  const value = {
    user: isSuccess ? user : null,
    isAuthenticated: !!localToken && isSuccess,
    isLoading: isLoading && !!localToken,
    login,
    // ...logout, signup
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
```

This centralizes all authentication logic, keeping components clean and focused on their primary function.
