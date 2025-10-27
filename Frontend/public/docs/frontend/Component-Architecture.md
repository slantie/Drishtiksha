# Frontend Component Architecture

## 1. Overview

The frontend's component architecture is based on a **Design System** approach, ensuring a consistent, professional, and maintainable user interface. We follow a philosophy similar to **Atomic Design**, where the UI is built from small, reusable pieces that are assembled into larger, more complex components and pages.

- **UI Primitives (`/ui`):** These are the basic building blocks, like `Button`, `Card`, and `Input`. They define the core look and feel of the application.
- **Business Components (`/components`):** These are more complex components that solve specific business problems, such as the `MediaPlayer` or the `AnalysisInProgress` panel. They are built by combining UI primitives.
- **Pages (`/pages`):** These are the final, user-facing views, which arrange business components into complete application screens like the `Dashboard` or `Results` page.

This layered approach allows for rapid development, ensures visual consistency, and makes the application easy to update and maintain.

---

## 2. Technical Deep Dive

### 2.1. Directory Structure

The component hierarchy is strictly enforced through the directory structure within `/src/components`.

- **`/src/components/ui` - The Design System:**

  - **Purpose:** Contains low-level, highly reusable, and "dumb" UI components. These components know nothing about business logic or API data. They are purely concerned with presentation and styling.
  - **Examples:** `Button.jsx`, `Card.jsx`, `DataTable.jsx`, `Modal.jsx`, `Input.jsx`.
  - **Technology:** Built using **Radix UI** primitives for accessibility and styled with **`class-variance-authority` (cva)** and **Tailwind CSS** for themeable, variant-based styling.

- **`/src/components/` - Business & Feature Components:**

  - **Purpose:** Contains "smart" components that encapsulate a specific feature or piece of business logic. They often fetch data using custom hooks and manage their own state.
  - **Examples:**
    - `MediaPlayer.jsx`: Manages video playback state.
    - `UploadModal.jsx`: Handles file selection, validation, and upload mutation.
    - `AnalysisInProgress.jsx`: Subscribes to real-time progress updates.
    - `charts/`: A sub-directory containing all data visualization components built with **Recharts**.

- **`/src/pages/` - Page-Level Components:**
  - **Purpose:** The final layer that assembles layout and business components into a complete view. These components are directly mapped to routes in `App.jsx`.
  - **Responsibilities:** Fetching the primary data for the page, handling page-level state (like modals), and passing data down to child components.
  - **Examples:** `Dashboard.jsx`, `Results.jsx`, `Profile.jsx`.

### 2.2. The UI Design System (`/ui`)

The UI library is built on a modern, flexible foundation.

#### Styling with CVA and Tailwind Merge

We use `class-variance-authority` (cva) to create type-safe, variant-based components. This allows a single component like `Button` to have multiple styles (`variant`, `size`) that can be mixed and matched.

**Example: `Button.jsx` Variants**

```javascript
// src/components/ui/Button.jsx
import { cva } from "class-variance-authority";

const buttonVariants = cva(
  "inline-flex items-center justify-center ...", // Base styles
  {
    variants: {
      variant: {
        default: "bg-primary-main text-white hover:bg-primary-main/90",
        destructive: "bg-red-600/25 text-red-500 hover:bg-red-600/40",
        outline: "border border-input bg-transparent hover:bg-accent",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-9 px-3",
        lg: "h-11 px-8",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);
```

The `cn` utility (`lib/utils.js`) then merges these generated classes with any custom classes passed via `className`, using `tailwind-merge` to intelligently resolve conflicting Tailwind classes.

```javascript
// Merges classes without conflicts
cn(buttonVariants({ variant: "outline" }), "mt-4");
// -> "inline-flex ... border border-input ... mt-4"
```

#### Key UI Components

- **`Card.jsx`**: The fundamental layout primitive for all content sections. Provides a consistent border, background, and padding structure.
- **`DataTable.jsx`**: A powerful, reusable table component with built-in sorting, filtering, pagination, and loading/empty states.
- **`Modal.jsx`**: A fully accessible, animated modal component built with Framer Motion for all pop-up dialogs.
- **`Input.jsx`**: A flexible input component that supports icons, error states, and different types.
- **`Badge.jsx`**: A versatile component for displaying status labels (`ANALYZED`, `PROCESSING`), media types, and other tags with consistent, themeable colors.

### 2.3. Layout Components (`/layout`)

These components define the persistent structure of the application.

- **`Layout.jsx`**: The main application shell. It renders the `Header`, `Footer`, and a `main` content area. All protected pages are wrapped in this component.
- **`AuthLayout.jsx`**: A specialized, centered layout used exclusively for the login and signup forms.
- **`Header.jsx`**: Contains the main navigation, user profile dropdown (or login/signup buttons), and the theme toggle.
- **`PageHeader.jsx`**: A standardized header used at the top of every main page, providing a consistent title, description, and action button area.

### 2.4. Business Components (`/components`)

These components are the workhorses of the application.

#### **`MediaPlayer.jsx`**

- **Responsibility:** Provides a fully-featured media player for video and audio files, with custom controls for play/pause, volume, seeking, playback speed, and fullscreen mode.
- **State Management:** Uses a `useReducer` hook internally to manage its complex state (isPlaying, currentTime, volume, etc.) in a predictable way.
- **Keyboard Shortcuts:** Implements keyboard controls (spacebar for play/pause, 'f' for fullscreen) for enhanced accessibility and user experience.

#### **`UploadModal.jsx`**

- **Responsibility:** Manages the entire file upload workflow.
- **Features:**
  - Drag-and-drop and file browser support.
  - Client-side validation for file type and size to provide instant feedback and prevent unnecessary backend requests.
  - Uses the `useUploadMediaMutation` hook from React Query to handle the API call and loading state.
  - On successful upload, triggers navigation to the new results page.

#### **`AnalysisInProgress.jsx` and `ProgressPanel.jsx`**

- **Responsibility:** Provides the real-time progress tracking experience.
- **`AnalysisInProgress.jsx`:** A high-level component that uses the `useAnalysisProgress` hook to get live data. It displays an overall progress summary and a list of `ModelStatusRow` components.
- **`ProgressPanel.jsx`:** A detailed, floating panel that shows granular, TQDM-style progress for each model, including stats like items per second and estimated time remaining.

#### **Chart Components (`/components/analysis/charts`)**

- **Responsibility:** A suite of modular components for data visualization, all built with **Recharts**.
- **Examples:**
  - `ModelConfidenceChart.jsx`: Bar chart comparing model confidence scores.
  - `FrameAnalysisChart.jsx`: Line chart showing frame-by-frame suspicion scores over time.
  - `PredictionDistributionChart.jsx`: Pie chart showing the ratio of "REAL" vs. "FAKE" predictions.
- **Design:** Each chart is a self-contained component that accepts data via props and handles its own rendering, including empty/loading states.
