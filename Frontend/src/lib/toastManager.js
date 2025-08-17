// src/lib/toastManager.js

class ToastManager {
    videoToastMap = new Map();

    register(videoId, toastId) {
        this.videoToastMap.set(videoId, toastId);
    }

    get(videoId) {
        return this.videoToastMap.get(videoId);
    }

    unregister(videoId) {
        this.videoToastMap.delete(videoId);
    }
}

export const toastManager = new ToastManager();
