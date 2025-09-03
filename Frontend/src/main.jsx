// src/main.jsx

import "./index.css";
import App from "./App.jsx";
import ReactDOM from "react-dom/client";
import { QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import queryClient from "./lib/queryClient.js";

// Set document title from environment variables
const projectName = import.meta.env.VITE_PROJECT_NAME || "Drishtiksha";
const projectDesc = import.meta.env.VITE_PROJECT_DESC || "Deepfake Detection";
document.title = `${projectName} - ${projectDesc}`;

ReactDOM.createRoot(document.getElementById("root")).render(
  <QueryClientProvider client={queryClient}>
    <App />
    <ReactQueryDevtools initialIsOpen={false} />
  </QueryClientProvider>
);
