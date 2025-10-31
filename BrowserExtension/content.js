// File: BrowserExtension/content.js

// This script is injected into the Drishtiksha web application page.
// Optional: Can sync cookie/localStorage if needed, but not required for core flow.

// Listen for messages from the background script (if switched to messaging).
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === "GET_AUTH_TOKEN") {
    console.log("Content Script: Received GET_AUTH_TOKEN request from background.");
    
    // Direct localStorage access as fallback
    const token = localStorage.getItem('authToken');
    if (token) {
      sendResponse({ token: token });
      return;
    }
    
    // PostMessage fallback if page has a listener
    const handlePageResponse = (event) => {
      if (event.source === window && event.data.type === "AUTH_TOKEN_RESPONSE") {
        console.log("Content Script: Received token via postMessage.");
        sendResponse({ token: event.data.token });
        window.removeEventListener("message", handlePageResponse);
      }
    };
    
    window.addEventListener("message", handlePageResponse);
    window.postMessage({ type: "REQUEST_AUTH_TOKEN" }, "*");
    
    return true; // Async response
  }
  
  return false;
});