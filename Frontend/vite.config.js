// vite.config.js

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";

export default defineConfig({
    plugins: [react()],
    // This server block is essential for SPA routing during development
    server: {
        // middlewareMode: false is the default, but appType implies it.
        // This ensures that Vite handles history API fallback for SPA routing.
        appType: "spa",
    },
    test: {
        globals: true,
        environment: "jsdom",
        setupFiles: "./src/tests/setup.js",
        css: true,
        include: ["**/*.{test,spec,integration.test}.{js,jsx}"],
    },
});
