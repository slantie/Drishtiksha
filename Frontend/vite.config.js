// vite.config.js

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";

export default defineConfig({
    plugins: [react()],
    test: {
        globals: true,
        // Use 'jsdom' for component tests, but we'll override for our new API tests
        environment: "jsdom",
        setupFiles: "./src/tests/setup.js",
        css: true,
        // Add this to include our new test file pattern
        include: ["**/*.{test,spec,integration.test}.{js,jsx}"],
    },
});
