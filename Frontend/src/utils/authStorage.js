// utils/authStorage.js

const TOKEN_KEY = "authToken";
const USER_KEY = "user";
const REMEMBER_KEY = "rememberMe";

export const authStorage = {
    set({ token, user, rememberMe }) {
        const storage = rememberMe ? localStorage : sessionStorage;
        const otherStorage = rememberMe ? sessionStorage : localStorage;

        try {
            // Store in the appropriate storage
            storage.setItem(TOKEN_KEY, token);
            storage.setItem(USER_KEY, JSON.stringify(user));
            storage.setItem(REMEMBER_KEY, rememberMe.toString());

            // Clear from the other storage to avoid conflicts
            otherStorage.removeItem(TOKEN_KEY);
            otherStorage.removeItem(USER_KEY);
            otherStorage.removeItem(REMEMBER_KEY);
        } catch (error) {
            console.error("Failed to store auth data:", error);
            // Fallback to sessionStorage if localStorage fails
            if (rememberMe) {
                sessionStorage.setItem(TOKEN_KEY, token);
                sessionStorage.setItem(USER_KEY, JSON.stringify(user));
                sessionStorage.setItem(REMEMBER_KEY, "false");
            }
        }
    },

    get() {
        try {
            // Check localStorage first (remember me), then sessionStorage
            const localToken = localStorage.getItem(TOKEN_KEY);
            const sessionToken = sessionStorage.getItem(TOKEN_KEY);

            const token = localToken || sessionToken;
            const storage = localToken ? localStorage : sessionStorage;

            const userStr = storage.getItem(USER_KEY);
            const rememberMe = storage.getItem(REMEMBER_KEY) === "true";

            return {
                token,
                user: userStr ? JSON.parse(userStr) : null,
                rememberMe,
            };
        } catch (error) {
            console.error("Failed to retrieve auth data:", error);
            return { token: null, user: null, rememberMe: false };
        }
    },

    clear() {
        try {
            // Clear ALL auth data from both storages
            localStorage.removeItem(TOKEN_KEY);
            localStorage.removeItem(USER_KEY);
            localStorage.removeItem(REMEMBER_KEY);
            sessionStorage.removeItem(TOKEN_KEY);
            sessionStorage.removeItem(USER_KEY);
            sessionStorage.removeItem(REMEMBER_KEY);

            // Note: User preferences (userPreferences.js) are stored separately
            // and will not be affected by this auth storage clear
        } catch (error) {
            console.error("Failed to clear auth data:", error);
        }
    },

    isExpired() {
        const { token } = this.get();
        if (!token) return true;

        try {
            // Decode JWT to check expiration (basic check)
            const payload = JSON.parse(atob(token.split(".")[1]));
            const now = Date.now() / 1000;
            return payload.exp < now;
        } catch (error) {
            console.error("Failed to parse token:", error);
            return true;
        }
    },
};
