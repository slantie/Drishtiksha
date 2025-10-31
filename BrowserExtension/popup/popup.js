// File: BrowserExtension/popup/popup.js

document.addEventListener('DOMContentLoaded', () => {
    const statusIndicator = document.getElementById('status-indicator');
    const statusText = document.getElementById('status-text');
    const loginPrompt = document.getElementById('login-prompt');
    const dashboardLink = document.getElementById('dashboard-link');

    if (!statusIndicator || !statusText || !loginPrompt || !dashboardLink) {
        console.error('Required DOM elements not found');
        return;
    }

    // Set the dashboard link URL
    dashboardLink.href = 'http://192.168.1.51:5173/dashboard';

    // Function to update UI based on auth status
    const updateStatus = (isLoggedIn, error = null) => {
        if (!statusIndicator || !statusText) {
            console.error('DOM elements not available for status update');
            return;
        }

        statusIndicator.className = `status-indicator ${isLoggedIn ? 'logged-in' : 'logged-out'}`;
        
        if (error) {
            statusText.textContent = error;
            statusText.className = 'status-text error';
            loginPrompt.style.display = 'block';
        } else if (isLoggedIn) {
            statusText.textContent = '✅ Logged in. Ready to analyze media!';
            statusText.className = 'status-text success';
            loginPrompt.style.display = 'none';
        } else {
            statusText.textContent = '❌ Not logged in.';
            statusText.className = 'status-text error';
            loginPrompt.style.display = 'block';
        }
    };

    // Check auth status using cookies (no tabs or scripting needed)
    const checkAuth = () => {
      chrome.cookies.get({ url: 'http://192.168.1.51:5173', name: 'authToken' }, (cookie) => {
        if (chrome.runtime.lastError) {
          console.error('Cookie check failed:', chrome.runtime.lastError);
          updateStatus(false, 'Failed to check login status. Please refresh the extension.');
        } else {
          updateStatus(!!cookie);
        }
      });
    };

    // Initial check
    checkAuth();

    // Optional: Poll every 30s for status changes (if user logs in/out in another tab)
    setInterval(checkAuth, 30000);
});