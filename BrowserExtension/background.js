// File: BrowserExtension/background.js

const BACKEND_URL = "http://192.168.1.51:3000";
const FRONTEND_URL = "http://192.168.1.51:5173";
const NOTIFICATION_ICON = chrome.runtime.getURL("icons/Icon-512.png"); // Full runtime URL

let analysisNotificationId = null;

// --- 1. INSTALLATION: CREATE THE CONTEXT MENU ---
chrome.runtime.onInstalled.addListener(() => {
  console.log("Drishtiksha Extension Installed/Updated.");
  chrome.contextMenus.create({
    id: "analyzeWithDrishtiksha",
    title: "Analyze with Drishtiksha",
    contexts: ["image", "video", "audio"],
  });

  // Clear any stale notifications on install
  chrome.notifications.getAll((notifications) => {
    Object.keys(notifications).forEach((id) => {
      if (id.startsWith("drishtiksha-")) {
        chrome.notifications.clear(id, () => {});
      }
    });
  });
});

// --- 2. EVENT LISTENER: HANDLE CONTEXT MENU CLICKS ---
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "analyzeWithDrishtiksha") {
    const mediaUrl = info.srcUrl;
    if (!mediaUrl) {
      showNotification(
        "error",
        "No media URL found",
        "Please right-click on a valid media element."
      );
      return;
    }

    showNotification(
      "info",
      "Analysis Started",
      "Uploading media for deepfake analysis. Visit your dashboard for results!"
    );

    try {
      // 1. Get the auth token from cookies (no tabs needed)
      const authToken = await getAuthToken();

      // 2. Fetch the media file as a Blob.
      const { blob: mediaBlob, filename, size } = await fetchMedia(mediaUrl);

      // Check file size limit (e.g., 100MB fallback)
      if (size > 100 * 1024 * 1024) {
        throw new Error(
          "Media file too large (max 100MB). Please try a smaller file."
        );
      }

      // 3. Post the media to your backend API.
      const responseData = await postToBackend(
        mediaBlob,
        filename,
        mediaUrl,
        authToken
      );

      showNotification(
        "success",
        "Upload Complete",
        `"${responseData.filename}" queued for analysis. Check dashboard for results.`
      );
    } catch (error) {
      console.error("Drishtiksha analysis failed:", error);
      showNotification(
        "error",
        "Analysis Failed",
        `Could not start analysis: ${error.message}`
      );
    }
  }
});

// --- HELPER FUNCTIONS ---

/**
 * Shows a simple notification with fallback if icon fails.
 */
function showNotification(type, title, message) {
  const notificationId = `drishtiksha-${Date.now()}`;

  const options = {
    type: "basic",
    title: title,
    message: message,
  };

  // Try with icon first
  if (NOTIFICATION_ICON) {
    options.iconUrl = NOTIFICATION_ICON;
  }

  // Optional: Add buttons for user interaction (e.g., open dashboard)
  if (type === "success") {
    options.buttons = [{ title: "View Dashboard" }];
  }

  chrome.notifications.create(notificationId, options, () => {
    if (chrome.runtime.lastError) {
      console.error(
        "Notification creation failed:",
        chrome.runtime.lastError.message || chrome.runtime.lastError
      );

      // Fallback: Retry without icon if it's an image download error
      if (
        chrome.runtime.lastError.message &&
        chrome.runtime.lastError.message.includes("images")
      ) {
        console.log(
          "[Background Script] Retrying notification without icon..."
        );
        const fallbackOptions = { ...options };
        delete fallbackOptions.iconUrl;

        chrome.notifications.create(
          notificationId + "-fallback",
          fallbackOptions,
          () => {
            if (chrome.runtime.lastError) {
              console.error(
                "Fallback notification also failed:",
                chrome.runtime.lastError.message || chrome.runtime.lastError
              );
            } else {
              console.log(
                `[Background Script] Fallback notification created: ${notificationId}-fallback`
              );
            }
          }
        );
      }
    } else {
      console.log(
        `[Background Script] Notification created: ${notificationId}`
      );
    }
  });

  // Auto-clear after 6s (longer for success to allow button interaction)
  const clearDelay = type === "success" ? 8000 : 6000;
  setTimeout(() => {
    chrome.notifications.clear(notificationId, () => {
      if (chrome.runtime.lastError) {
        console.error("Notification clear failed:", chrome.runtime.lastError);
      }
    });
    // Also clear fallback if created
    chrome.notifications.clear(notificationId + "-fallback", () => {});
  }, clearDelay);

  // Handle button clicks (e.g., open dashboard) - listener is global but IDs are unique
  chrome.notifications.onButtonClicked.addListener(
    (clickedNotificationId, buttonIndex) => {
      if (
        (clickedNotificationId === notificationId ||
          clickedNotificationId === notificationId + "-fallback") &&
        buttonIndex === 0
      ) {
        chrome.tabs.create({ url: FRONTEND_URL + "/dashboard" });
      }
    }
  );
}

/**
 * Gets the auth token directly from cookies (no tabs or scripting needed).
 */
async function getAuthToken() {
  return new Promise((resolve, reject) => {
    chrome.cookies.get({ url: FRONTEND_URL, name: "authToken" }, (cookie) => {
      if (chrome.runtime.lastError) {
        console.error("Cookie access error:", chrome.runtime.lastError);
        reject(
          new Error(
            "Cookie access failed. Ensure cookie permissions are granted."
          )
        );
      } else if (!cookie || !cookie.value) {
        reject(
          new Error(
            "No auth token found in cookies. Please log in to Drishtiksha."
          )
        );
      } else {
        console.log(
          "[Background Script] Token retrieved from cookies successfully"
        );
        resolve(cookie.value);
      }
    });
  });
}

/**
 * Fetches media with better error handling and size tracking.
 */
async function fetchMedia(url) {
  try {
    if (url.startsWith("data:")) {
      console.log("[Background Script] Processing base64 data URL...");
      return handleDataUrl(url);
    }

    console.log("[Background Script] Fetching from URL:", url);
    const response = await fetch(url, {
      method: "GET",
      mode: "cors",
      cache: "default",
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const blob = await response.blob();
    const filename = extractFilenameFromUrl(url, blob.type);
    const size = blob.size;

    return { blob, filename, size };
  } catch (error) {
    console.error("Error fetching media:", error);
    if (error.message.includes("cors") || error.message.includes("network")) {
      throw new Error(
        "Failed to fetch media (CORS/Network issue). Try saving the file and uploading manually."
      );
    }
    throw new Error(`Failed to fetch media: ${error.message}`);
  }
}

/**
 * Converts data URL to Blob with error handling.
 */
function handleDataUrl(dataUrl) {
  try {
    const matches = dataUrl.match(/^data:([^;]+);base64,(.+)$/);
    if (!matches) {
      throw new Error("Invalid data URL format");
    }

    const mimeType = matches[1];
    const base64Data = matches[2];

    if (base64Data.length > 100 * 1024 * 1024 * 1.33) {
      throw new Error("Data URL too large (max 100MB)");
    }

    const binaryString = atob(base64Data);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }

    const blob = new Blob([bytes], { type: mimeType });
    const extension = getExtensionFromMimeType(mimeType);
    const filename = `media-${Date.now()}${extension}`;

    console.log(
      `[Background Script] Converted data URL: ${filename} (${blob.size} bytes)`
    );
    return { blob, filename, size: blob.size };
  } catch (error) {
    console.error("Data URL processing error:", error);
    throw new Error(`Failed to process data URL: ${error.message}`);
  }
}

/**
 * Extracts or generates filename.
 */
function extractFilenameFromUrl(url, mimeType) {
  try {
    const urlObj = new URL(url);
    const urlFilename = urlObj.pathname.split("/").pop();
    if (urlFilename && urlFilename.includes(".")) {
      return urlFilename;
    }
  } catch (e) {
    console.warn("Invalid URL for filename extraction:", e);
  }

  const extension = getExtensionFromMimeType(mimeType);
  return `media-${Date.now()}${extension}`;
}

/**
 * MIME to extension mapping.
 */
function getExtensionFromMimeType(mimeType) {
  const mimeToExt = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/svg+xml": ".svg",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "video/x-msvideo": ".avi",
    "video/avi": ".avi",
    "video/x-matroska": ".mkv",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/ogg": ".ogg",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/webm": ".webm",
  };

  return mimeToExt[mimeType] || ".bin";
}

/**
 * Sends the media blob to your backend API using fetch.
 */
async function postToBackend(blob, filename, originalUrl, token) {
  const formData = new FormData();

  formData.append("media", blob, filename);
  formData.append(
    "description",
    `Analyzed from Drishtiksha browser extension: ${originalUrl.substring(
      0,
      100
    )}...`
  );

  const response = await fetch(`${BACKEND_URL}/api/v1/media`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
    },
    body: formData,
  });

  let responseData;
  try {
    responseData = await response.json();
  } catch (parseError) {
    throw new Error("Invalid JSON response from backend");
  }

  if (!response.ok) {
    throw new Error(responseData.message || "Backend API returned an error.");
  }

  if (!responseData.data) {
    throw new Error(responseData.message || "Unexpected response format");
  }

  return responseData.data;
}
