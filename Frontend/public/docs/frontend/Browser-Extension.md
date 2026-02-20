# Drishtiksha Browser Extension

## 1. Overview

The **Drishtiksha Browser Extension** is a productivity tool designed to seamlessly integrate our powerful deepfake detection capabilities directly into the user's daily web browsing workflow. It eliminates the cumbersome process of manually downloading and re-uploading media, transforming deepfake analysis from a multi-step task into a simple, two-click action.

By leveraging the user's existing Drishtiksha session, the extension provides a secure and frictionless way to send any image, video, or audio file from any website directly to our analysis pipeline. This makes proactive media verification an effortless part of any workflow, from journalism and research to content moderation.

### 1.1. Key Features

- **One-Click Analysis:** Right-click any media on any webpage and select "Analyze with Drishtiksha" to instantly queue it for analysis.
- **Seamless Authentication:** The extension securely uses the active login session from the main Drishtiksha web application. There are no separate logins required.
- **Instant Feedback:** Users receive immediate desktop notifications confirming that their analysis has started, and another notification upon completion.
- **Centralized Results:** All analyses initiated from the extension are automatically saved to the user's central Drishtiksha dashboard for review and report generation.
- **Lightweight & Secure:** The extension runs as an efficient service worker (Manifest V3) and communicates securely with our existing, trusted backend API.

### 1.2. The Value Proposition: From Chore to Reflex

In an environment saturated with AI-generated content, the speed of verification is critical. The browser extension transforms deepfake detection from a disruptive chore into an instant reflex.

| Manual Workflow (Without Extension)                    | Extension Workflow                                    |
| :----------------------------------------------------- | :---------------------------------------------------- |
| 1. Find suspicious media.                              | 1. Find suspicious media.                             |
| 2. Right-click and save the file to disk.              | 2. Right-click and select "Analyze with Drishtiksha". |
| 3. Open a new tab and navigate to the Drishtiksha app. |                                                       |
| 4. Log in.                                             |                                                       |
| 5. Drag and drop the downloaded file to upload.        |                                                       |
| 6. Wait for the analysis to complete.                  |                                                       |
| **Time Taken: 1-2 minutes per file.**                  | **Time Taken: < 5 seconds per file.**                 |

This dramatic increase in efficiency empowers users to verify media at the speed they browse, making the Drishtiksha platform an indispensable tool for daily work.

---

## 2. Technical Architecture & Data Flow

The extension is architected as a lightweight client that acts as a secure bridge between the user's browser and the existing Drishtiksha backend infrastructure. **No changes to the backend or ML server are required.**

### 2.1. High-Level Diagram

```text
┌────────────────┐   Right-Click     ┌─────────────────┐   Message   ┌───────────────────┐
│ Any Web Page   ├─────────────────► │ background.js   ├────────────►│    content.js     │
│ (e.g., cnn.com)│  (Capture srcUrl) │ (Service Worker)│ (Request)   │ (Injected Script) │
└────────────────┘                   └────────┬────────┘             └──────────┬────────┘
                                              │                                 │
           ┌──────────────────────────────────┘                                 │ Message
           │ 4. POST /api/v1/media (with Token & Blob)                          │ (Response)
           ▼                                                                    │
┌────────────────┐   localStorage   ┌─────────────────┐                         │
│ Drishtiksha    │◄─────────────────┤ React Web App   │◄────────────────────────┘
│ Backend API    │   (Read Token)   │ (http://192...) │   3. Get Token from
└────────────────┘                  └─────────────────┘      localStorage
```

### 2.2. The End-to-End Workflow

1.  **Context Menu Trigger (`background.js`):** When a user right-clicks on a media element, the `chrome.contextMenus.onClicked` listener in the background script is triggered. It captures the `srcUrl` of the media.

2.  **Authentication via Message Passing:**

    - The `background.js` script **cannot** directly access the `localStorage` of your web app due to browser security policies.
    - Instead, it sends a message (`{ type: "GET_AUTH_TOKEN" }`) to the `content.js` script, which is injected _only_ into the pages of your active Drishtiksha web application.
    - The `content.js` script receives this message and forwards it to the web page's `window`.
    - Your React app (`App.jsx`) has a listener that hears this message, retrieves the `authToken` from `authStorage.js`, and posts it back to the `content.js` script.
    - The `content.js` script relays the token back to `background.js`.

3.  **Media Fetching (`background.js`):** The background script uses the `fetch()` API to download the media from its `srcUrl` and convert it into a `Blob`. This is necessary because the backend API expects a file upload, not just a URL.

4.  **API Call (`background.js` -> Backend):**

    - The background script constructs a `FormData` object, appending the media `Blob` and a description.
    - It then makes a `POST` request to the existing `/api/v1/media` endpoint on your backend.
    - Crucially, it includes the JWT obtained via message passing in the `Authorization: Bearer <token>` header.

5.  **Backend Processing (Existing Infrastructure):** Your backend receives the request and treats it identically to an upload from the main web app. It validates the token, stores the file, queues the job in BullMQ, and begins the asynchronous analysis pipeline.

6.  **User Feedback (`background.js`):** The background script uses the `chrome.notifications` API to display notifications to the user, confirming that the analysis has been successfully queued. Further real-time updates are available on the main dashboard.

---

## 3. Component Breakdown

The extension is composed of several distinct files, each with a specific role.

### 3.1. `manifest.json`

This is the configuration and entry point of the extension.

- **`"manifest_version": 3`**: Declares that this is a modern, secure Manifest V3 extension.
- **`"background": { "service_worker": "background.js" }`**: Registers the background script as a non-persistent service worker, which is more memory-efficient than a persistent background page.
- **`"permissions": [...]`**:
  - `contextMenus`: To create the right-click menu.
  - `notifications`: To show pop-up alerts.
  - `storage`: For the extension to store its own settings (optional).
  - `cookies`: **This is a fallback/alternative authentication method and is not used in the primary message-passing flow.**
- **`"content_scripts": [...]`**: This is a critical section. It tells Chrome to inject `content.js` **only** into pages matching your web app's URL (`http://localhost:5173/*`).
- **`"host_permissions": [...]`**: Grants the extension the necessary network access:
  - `"http://localhost/*"`: Allows the content script to be injected.
  - `"http://localhost:3000/*"`: Allows `background.js` to make API calls to your backend.
  - `"<all_urls>"`: Allows `background.js` to `fetch` media from any website on the internet.

### 3.2. `background.js` (The Core Logic)

This service worker is the brain of the extension.

- **`chrome.runtime.onInstalled`**: This listener runs once when the extension is installed or updated. It sets up the `Analyze with Drishtiksha` context menu item.
- **`chrome.contextMenus.onClicked`**: This is the main event handler. It orchestrates the entire workflow when a user clicks the menu item.
- **`getAuthTokenFromWebApp()`**: This function encapsulates the message-passing logic. It finds the active Drishtiksha tab and sends a message to its content script, waiting for the token in response. This is the secure bridge for authentication.

### 3.3. `content.js` (The Injectable Bridge)

This small script is the key to bypassing the browser's security sandbox.

- **Role:** It acts as a secure message forwarder. It has access to both the extension's message system (`chrome.runtime.onMessage`) and the web page's message system (`window.postMessage`).
- **Workflow:**
  1.  Receives `GET_AUTH_TOKEN` from `background.js`.
  2.  Posts `REQUEST_AUTH_TOKEN` to the web page `window`.
  3.  Listens for the `AUTH_TOKEN_RESPONSE` from the web page.
  4.  Forwards the token in that response back to `background.js`.

### 3.4. `App.jsx` Modifications (The Web App's Role)

Your React application is made an active participant in this flow by adding a simple `useEffect` hook.

- **`window.addEventListener("message", ...)`**: This listener allows your web app to securely respond to requests from its own injected content script.
- **Security:** It checks `event.source === window` to ensure it's not responding to messages from other browser tabs or iframes, and `event.data.type === "REQUEST_AUTH_TOKEN"` to ensure it's the correct request.

### 3.5. `popup.html` & `popup.js` (The Toolbar UI)

This provides a simple status window when the user clicks the extension icon.

- **Functionality:** It checks for the existence of the authentication token (using the same message-passing or cookie method) and displays whether the user is logged in or not, providing a quick link to the main dashboard. It serves as a simple status indicator and entry point.
