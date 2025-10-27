# Frontend Real-Time System Architecture

## 1. Overview

The real-time system is the core of the Drishtiksha frontend's user experience. It is designed to give users **instant, granular feedback** on their analysis jobs, which can take several minutes to complete. This prevents the user from feeling like the application is frozen and provides transparency into the complex work happening behind the scenes.

**The User Experience:**

1.  A user uploads a video.
2.  A persistent **Progress Panel** appears, showing the overall status.
3.  The panel live-updates as each individual AI model starts, processes, and completes its analysis.
4.  Once all analyses are finished, a final "Complete" or "Failed" notification is shown, and the panel automatically closes.

This entire process is managed by a decoupled, event-driven system using **WebSockets**, a custom **Toast Orchestrator**, and specialized **React Hooks**.

---

## 2. Technical Deep Dive

### 2.1. System Architecture & Event Flow

The real-time system is a closed loop that connects the backend workers to the user's browser, with the API server and Redis acting as intermediaries.

```text
┌────────────────┐   ┌─────────────────┐   ┌─────────────────┐   ┌────────────────┐
│ Backend Worker │──►│  Redis Pub/Sub  │──►│   API Server    │──►│ Frontend Client│
│ (BullMQ Job)   │   │ (media-progress)│   │ (Socket.IO Srv) │   │ (Socket.IO Lib)│
└────────────────┘   └─────────────────┘   └─────────────────┘   └────────────────┘
       │                      ▲
       │ 1. Publishes         │ 2. Listens for
       │    progress event    │    messages
       └──────────────────────┘
                              │                      ▲
                              │ 3. Emits event       │ 4. Receives event
                              │    to user's room    │    and updates UI
                              └──────────────────────┘
```

1.  **Publisher (Backend Worker):** As a worker processes a job (e.g., analyzing frames), it publishes `ProgressEvent` objects to a specific Redis channel (`media-progress-events`). It is a "fire-and-forget" operation.
2.  **Broker (Redis):** Redis immediately broadcasts this event to all subscribed clients.
3.  **Subscriber (API Server):** The Node.js API server has a persistent Redis client subscribed to the channel. When it receives an event, it looks up the `userId` from the event payload.
4.  **Broadcaster (Socket.IO Server):** The API server then emits a `progress_update` event into a private, user-specific Socket.IO "room."
5.  **Listener (Frontend Client):** The user's browser, connected to the Socket.IO server, receives this event and updates the UI accordingly.

### 2.2. Socket.IO Client

This module exports a singleton `socketService` instance that manages the client-side WebSocket connection.

- **Connection Management:**
  - `connect(token)`: Establishes a connection to the backend, passing the user's JWT for authentication. Called by `AuthContext` on login.
  - `disconnect()`: Closes the connection. Called by `AuthContext` on logout.
- **Event Handling:**
  - It listens for three primary server-pushed events:
    1.  **`progress_update`**: Fired for granular updates during analysis (e.g., model started, frame progress).
    2.  **`media_update`**: Fired once when an entire `AnalysisRun` is complete, providing the final, updated `Media` object.
    3.  **`processing_error`**: Fired if a job fails irrecoverably.
  - **Crucially, it does not update UI state directly.** Instead, it delegates all incoming events to the `ToastOrchestrator`.

```javascript
// src/lib/socket.jsx (Simplified)
class SocketService {
  socket = null;

  connect(token) {
    this.socket = io(config.VITE_BACKEND_URL, { auth: { token } });

    this.socket.on("connect", () => console.log("Socket connected!"));
    this.setupEventListeners();
  }

  setupEventListeners() {
    // For every progress event, delegate to the orchestrator.
    this.socket.on("progress_update", (progress) => {
      toastOrchestrator.handleProgressEvent(
        progress.mediaId,
        progress.event,
        progress.message,
        progress.data
      );
    });

    // For the final media update, resolve the process.
    this.socket.on("media_update", (media) => {
      toastOrchestrator.resolveMediaProcessing(media.id, media.filename, true);
      // Also update React Query cache directly.
      queryClient.setQueryData(queryKeys.media.detail(media.id), {
        data: media,
      });
    });
  }
}
```

### 2.3. The Toast Orchestrator

This is a custom, non-React singleton class responsible for intelligently managing UI notifications to prevent "toast spam."

- **Problem:** A single analysis run can generate dozens of progress events. Showing a new toast for each event would overwhelm the user.
- **Solution:** The orchestrator maintains a map of `mediaId` to a single, persistent toast ID.

**Workflow:**

1.  **`handleProgressEvent`:** This is the main entry point, called by the `socketService`.
2.  **`startMediaProcessing`:** On the first event for a new `mediaId` (`PROCESSING_STARTED`), it creates a single, persistent `loading` toast that renders our custom `ToastProgress` component. It stores the ID of this toast.
3.  **`updateModelProgress`:** For all subsequent `progress_update` events for the same `mediaId`, it doesn't create a new toast. Instead, it uses `toast.loading()` on the _same toast ID_ to update the content of the existing toast.
4.  **`resolveMediaProcessing`:** When the final `media_update` event arrives (or an error), it dismisses the persistent loading toast and shows a final "Success" or "Error" message that auto-dismisses after a few seconds.

**In addition to managing toasts, it also acts as a pub/sub system for React hooks.**

- **`registerProgressCallback(mediaId, callback)`:** The `useAnalysisProgress` hook calls this to register its `setState` function.
- **Event Delegation:** When the orchestrator receives an event, it invokes the registered callback, passing the raw event data. This allows the `AnalysisProgress` component to update its detailed, multi-model view.

### 2.4. The `useAnalysisProgress` Hook

This custom hook is the bridge between the non-React `toastOrchestrator` and the React component tree.

- **Responsibility:**
  - Subscribes to the `toastOrchestrator` for a specific `mediaId`.
  - Manages the state for the detailed `AnalysisProgress` modal (`isProgressVisible`, `modelProgress`).
  - Calculates derived state like `overallProgress` and `isComplete`.

**Implementation:**

```javascript
// src/hooks/useAnalysisProgress.jsx (Simplified)
export const useAnalysisProgress = (mediaId, filename) => {
  const [isProgressVisible, setIsProgressVisible] = useState(false);
  const [modelProgress, setModelProgress] = useState({});

  // The callback that will be registered with the orchestrator.
  const handleProgressUpdate = useCallback((event) => {
    const { data } = event;
    if (data?.model_name) {
      // Update the state of this hook with the new progress data.
      setModelProgress((prev) => ({
        ...prev,
        [data.model_name]: { ...data },
      }));
    }

    if (event.event === "PROCESSING_STARTED") {
      setIsProgressVisible(true); // Auto-show the panel on start.
    }
  }, []);

  // Register and unregister the callback on mount/unmount.
  useEffect(() => {
    if (mediaId) {
      toastOrchestrator.registerProgressCallback(mediaId, handleProgressUpdate);
      return () => {
        toastOrchestrator.unregisterProgressCallback(mediaId);
      };
    }
  }, [mediaId, handleProgressUpdate]);

  // Expose state and control functions to the component.
  return {
    isProgressVisible,
    modelProgress,
    showProgress: () => setIsProgressVisible(true),
    hideProgress: () => setIsProgressVisible(false),
    // ...other derived values
  };
};
```

This architecture ensures a clean separation of concerns:

- `socketService`: Manages network connection.
- `toastOrchestrator`: Manages global notification state.
- `useAnalysisProgress`: Manages React component state for a specific media item.
