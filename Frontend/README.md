# Drishtiksha AI - Frontend Module ✅

## Overview

The frontend module is a modern React-based web application for the Drishtiksha AI Deepfake Detection System. It provides an intuitive user interface for video upload, analysis management, and comprehensive deepfake detection results visualization across multiple AI models. The application seamlessly integrates with the backend's asynchronous processing pipeline, providing real-time updates and comprehensive analysis results.

### 🎯 **Integration Status: PRODUCTION READY**

-   ✅ **Backend Integration**: Seamless communication with Node.js/Express backend
-   ✅ **Real-time Updates**: Live polling for analysis progress and results
-   ✅ **Multi-Model Support**: Displays results from all 3 AI models automatically
-   ✅ **Automated Workflow**: One-click upload triggers comprehensive analysis pipeline
-   ✅ **Manual Controls**: Granular model and analysis type selection for re-runs
-   ✅ **Responsive Design**: Optimized for desktop and mobile experiences
-   ✅ **Performance Optimized**: Fast loading with efficient state management

## Tech Stack

### Core Technologies

-   **React 19.1.0** - Modern React with latest features and performance optimizations
-   **Vite 6.3.5** - Fast build tool and development server
-   **React Router DOM 7.6.2** - Client-side routing for SPA navigation
-   **TypeScript Support** - Type-safe development environment

### UI/UX Framework

-   **Tailwind CSS 3.4.17** - Utility-first CSS framework for rapid UI development
-   **Lucide React 0.513.0** - Modern icon library with 1000+ icons
-   **Framer Motion 12.23.12** - Production-ready motion library for React
-   **Radix UI 1.4.2** - Low-level UI primitives for high-quality design systems
-   **Class Variance Authority** - Type-safe variant API for component styling

### State Management & Data Fetching

-   **TanStack React Query 5.85.0** - Powerful data synchronization for React
-   **React Query DevTools** - Development tools for debugging queries
-   **Axios 1.9.0** - Promise-based HTTP client for API communication

### Media & File Handling

-   **Cloudinary React 1.14.3** - Cloud-based image and video management
-   **React Player 3.3.1** - React component for playing videos
-   **Cloudinary Video Player 3.2.0** - Advanced video playback capabilities

### Additional Features

-   **React Hot Toast 2.5.2** - Lightweight toast notifications
-   **React CountUp 6.5.3** - Animated number counting component
-   **HTML2Canvas & jsPDF** - Client-side PDF report generation
-   **Prop Types 15.8.1** - Runtime type checking for React props

## Project Structure

```bash
Frontend/
├── public/       # Static assets
│ ├── Logo.png     # Application logo (PNG format)
│ └── Logo.svg     # Application logo (SVG format)
├── src/
│ ├── components/    # Reusable UI components
│ │ ├── ui/     # Base UI components (buttons, cards, etc.)
│ │ ├── layout/   # Layout components (header, footer, sidebar)
│ │ ├── forms/    # Form-related components
│ │ └── charts/   # Data visualization components
│ ├── pages/    # Page-level components
│ │ ├── Home.jsx  # Landing page
│ │ ├── Login.jsx   # User authentication
│ │ ├── Register.jsx  # User registration
│ │ ├── Dashboard.jsx # User dashboard
│ │ ├── Upload.jsx  # Video upload interface
│ │ ├── Results.jsx # Analysis results display
│ │ └── DetailedAnalysis.jsx # Comprehensive analysis view
│ ├── services/   # API service layer
│ │ ├── api.js    # Axios configuration and interceptors
│ │ ├── auth.service.js # Authentication API calls
│ │ ├── video.service.js # Video management API calls
│ │ └── analysis.service.js # Analysis-related API calls
│ ├── hooks/    # Custom React hooks
│ │ ├── useAuth.js  # Authentication state management
│ │ ├── useAnalysis.js # Analysis data management
│ │ └── useLocalStorage.js # Local storage utilities
│ ├── contexts/   # React context providers
│ │ ├── AuthContext.jsx # Authentication context
│ │ └── ThemeContext.jsx # Theme management
│ ├── providers/    # Higher-order providers
│ │ └── QueryProvider.jsx # React Query provider setup
│ ├── utils/    # Utility functions
│ │ ├── constants.js  # Application constants
│ │ ├── helpers.js  # Helper functions
│ │ └── validation.js # Form validation utilities
│ ├── constants/    # Application constants and configurations
│ ├── lib/      # Third-party library configurations
│ ├── App.jsx     # Root application component
│ ├── main.jsx    # Application entry point
│ └── index.css   # Global styles and Tailwind imports
├── eslint.config.js  # ESLint configuration
├── postcss.config.js   # PostCSS configuration
├── tailwind.config.js  # Tailwind CSS configuration
├── vite.config.js    # Vite build configuration
└── package.json    # Dependencies and scripts
```

## Key Features

### 🚀 **Automated Analysis Pipeline**

-   **One-Click Processing**: Upload triggers automatic analysis across all available models
-   **Real-time Progress**: Live updates with polling for analysis status and completion
-   **Comprehensive Results**: Displays QUICK, DETAILED, FRAMES, and VISUALIZE analyses automatically
-   **Background Processing**: Non-blocking UI with asynchronous backend integration

### 🎛️ **Advanced Analysis Controls**

-   **Multi-Model Support**: SIGLIP-LSTM-V1, SIGLIP-LSTM-V3, ColorCues-LSTM-V1 integration
-   **Manual Re-runs**: Granular control for re-running specific analysis types and models
-   **Version Tracking**: Complete history of analysis versions per video and model
-   **Model Selection**: Choose specific models for targeted analysis when needed

### 🔐 **Authentication & Security**

-   **JWT-based Authentication**: Secure user authentication with token management
-   **Protected Routes**: Route-level access control for authenticated users
-   **Session Management**: Automatic token refresh and logout handling
-   **Secure API Communication**: Token-based authentication for all backend requests

### 📊 **Rich Data Visualization**

-   **Interactive Charts**: Real-time visualization of analysis results and metrics
-   **Progress Tracking**: Visual progress indicators for ongoing analysis jobs
-   **Comprehensive Reports**: Detailed analysis breakdowns with charts and confidence scores
-   **Model Comparison**: Side-by-side comparison of results across different models

### 🎨 **Enhanced User Experience**

-   **Responsive Design**: Mobile-first approach with Tailwind CSS optimization
-   **Dark/Light Theme**: User-selectable theme preferences with system detection
-   **Intuitive Navigation**: Clear navigation with React Router and breadcrumbs
-   **Smart Notifications**: Context-aware toast notifications for actions and status updates
-   **Loading States**: Elegant loading indicators and skeleton screens

### 📱 **Advanced Media Management**

-   **Cloudinary Integration**: Cloud-based video storage with optimized delivery
-   **Video Player**: Advanced video playback with analysis overlays and annotations
-   **Drag-Drop Upload**: Intuitive file upload with validation and preview
-   **Thumbnail Generation**: Automatic video thumbnails and preview functionality
-   **Progress Streaming**: Real-time visualization video generation and display

## Environment Configuration

### Required Environment Variables

Create a `.env` file in the Frontend directory with the following production-ready configuration:

```env
# Backend API Configuration
VITE_API_BASE_URL=http://localhost:3000/api/v1
VITE_BACKEND_URL=http://localhost:3000

# Cloudinary Configuration (for video display)
VITE_CLOUDINARY_CLOUD_NAME=your_cloud_name
VITE_CLOUDINARY_API_KEY=your_api_key

# Application Configuration
VITE_APP_NAME="Drishtiksha AI"
VITE_APP_VERSION="2.0.0"
VITE_ENVIRONMENT="development"

# Feature Flags
VITE_ENABLE_ANALYTICS=true
VITE_ENABLE_DEV_TOOLS=true
VITE_ENABLE_AUTO_ANALYSIS=true
VITE_POLLING_INTERVAL=3000

# UI Configuration
VITE_MAX_FILE_SIZE=100
VITE_SUPPORTED_FORMATS="mp4,avi,mov,mkv"
VITE_RESULTS_PER_PAGE=10
```

### Integration-Specific Settings

**Backend Communication:**

-   `VITE_API_BASE_URL`: Full API endpoint for backend communication
-   `VITE_POLLING_INTERVAL`: Real-time update frequency (milliseconds)
-   `VITE_ENABLE_AUTO_ANALYSIS`: Toggle automatic analysis pipeline

**Performance Tuning:**

-   Smart polling intervals based on video status
-   Optimized React Query cache settings
-   Lazy loading for analysis visualization components

### Environment-Specific Configurations

**Development:**

-   Hot reload enabled
-   DevTools available
-   Detailed error messages
-   Source maps enabled

**Production:**

-   Optimized bundles
-   Minified assets
-   Error boundaries
-   Performance monitoring

## Installation & Setup

### Prerequisites

-   Node.js 18+ and npm/yarn
-   Backend API server running on port 3000
-   ML Server running on port 8000

### Quick Start

```bash
# Navigate to frontend directory
cd Frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Run linting
npm run lint
```

### Development Workflow

1. **Component Development**: Create reusable components in `src/components/`
2. **Page Creation**: Add new pages in `src/pages/` with proper routing
3. **API Integration**: Use services in `src/services/` for backend communication
4. **State Management**: Leverage React Query for server state and React hooks for local state
5. **Styling**: Use Tailwind CSS utilities with component-specific styles

## API Integration

### Simplified Workflow Architecture

The frontend now integrates with the backend's **asynchronous, queue-based architecture** for a streamlined user experience:

**Primary Workflow (Automatic):**

```javascript
// 1. Upload Video - Triggers automatic analysis
export const videoService = {
    uploadVideo: (formData) => api.post("/api/v1/videos", formData),
    // Returns: { id, status: "QUEUED", cloudinaryUrl, ... }

    // 2. Poll for Results - Real-time status updates
    getVideoDetails: (videoId) => api.get(`/api/v1/videos/${videoId}`),
    // Returns: { id, status: "PROCESSING|ANALYZED", analyses: [...] }
};
```

**Advanced Controls (Manual Re-runs):**

```javascript
// For granular control and re-analysis
export const analysisService = {
    // Trigger specific analysis (preserved for re-runs)
    triggerAnalysis: (videoId, analysisType, model) =>
        api.post(`/api/v1/videos/${videoId}/analyze`, { analysisType, model }),

    // Get detailed analysis results
    getAnalysisDetails: (analysisId) =>
        api.get(`/api/v1/analyses/${analysisId}`),

    // Check ML server health and available models
    getServerStatus: () => api.get("/api/v1/videos/status"),
};
```

### React Query Integration for Real-time Updates

```javascript
// Real-time video analysis tracking
export const useVideoAnalysis = (videoId) => {
    return useQuery({
        queryKey: ["video", videoId],
        queryFn: () => videoService.getVideoDetails(videoId),
        enabled: !!videoId,
        refetchInterval: (data) => {
            // Smart polling: faster when processing, slower when complete
            return data?.status === "PROCESSING"
                ? 3000
                : data?.status === "QUEUED"
                ? 5000
                : false;
        },
        refetchIntervalInBackground: true,
    });
};

// Server health monitoring
export const useServerHealth = () => {
    return useQuery({
        queryKey: ["serverHealth"],
        queryFn: () => analysisService.getServerStatus(),
        refetchInterval: 30000, // Check every 30 seconds
        retry: 3,
    });
};
```

## Component Architecture

### Design System Components

**Base Components:**

-   Button variants (primary, secondary, outline, ghost) with loading states
-   Card layouts (default, hover, interactive) with status indicators
-   Form inputs (text, file, select, textarea) with validation feedback
-   Modal dialogs and overlays with backdrop blur effects
-   Progress bars and circular indicators for real-time updates

**Composite Components:**

-   **VideoUpload**: Drag-drop interface with instant upload and queue status
-   **AnalysisCard**: Real-time result display with model badges and confidence scores
-   **ProgressIndicator**: Live progress tracking with time estimates
-   **DataTable**: Sortable, filterable results with export capabilities
-   **ModelSelector**: Advanced model and analysis type selection for re-runs

### Enhanced Page Components

**Upload.jsx Features:**

-   Drag-and-drop video upload with instant feedback
-   Automatic processing initiation with queue status
-   Real-time progress tracking from upload to analysis completion
-   Smart routing to results page when upload completes

**Results.jsx Features:**

-   **Automated Display**: Shows all analyses automatically as they complete
-   **Model Grouping**: Organized results by AI model (SIGLIP-LSTM-V1/V3, ColorCues-LSTM)
-   **Real-time Updates**: Live polling for analysis progress and new results
-   **Status Indicators**: Clear visual status (QUEUED → PROCESSING → ANALYZED)
-   **Interactive Cards**: Click to view detailed analysis breakdowns
-   **Re-run Controls**: Manual analysis triggers for specific models/types

**DetailedAnalysis.jsx Features:**

-   **Comprehensive Visualization**: Complete analysis breakdown with confidence metrics
-   **Frame-by-frame Analysis**: Timeline view with keyframe detection
-   **Temporal Charts**: Confidence evolution over video duration
-   **Model Comparison**: Side-by-side results from different models
-   **Visualization Player**: Embedded analysis visualization videos
-   **Export Options**: PDF reports and data export capabilities

**Dashboard.jsx Features:**

-   **Video Library**: Grid view of all user videos with analysis status
-   **Quick Actions**: One-click access to upload, view results, re-analyze
-   **Statistics Overview**: Analysis completion rates and model performance metrics
-   **Recent Activity**: Timeline of recent uploads and completed analyses

## User Workflow

### 🚀 **Streamlined Analysis Pipeline**

The frontend provides two complementary workflows optimized for different use cases:

#### **Primary Workflow: Automated Analysis**

1. **Upload Video**: Drag-and-drop or select video file
2. **Instant Processing**: Backend automatically queues comprehensive analysis
3. **Real-time Tracking**: Live progress updates across all models and analysis types
4. **Results Display**: Automatic display of completed analyses as they finish
5. **Comprehensive View**: Access to all 8+ analyses (QUICK, DETAILED, FRAMES, VISUALIZE) across 3 models

```
User Upload → Auto Queue → Real-time Updates → Complete Results
     ↓              ↓              ↓              ↓
   Upload.jsx → Results.jsx → Live Polling → DetailedAnalysis.jsx
```

#### **Advanced Workflow: Manual Control**

1. **Model Selection**: Choose specific AI models (SIGLIP-LSTM-V1/V3, ColorCues-LSTM)
2. **Analysis Type**: Select QUICK, DETAILED, FRAMES, or VISUALIZE analysis
3. **Targeted Processing**: Run specific combinations for comparison or re-analysis
4. **Custom Insights**: Generate focused results for research or debugging

```
Results.jsx → Model Selector → Custom Analysis → Targeted Results
     ↓              ↓              ↓              ↓
   Re-run UI → API Call → Background Processing → Updated Display
```

### 📊 **Real-time Status Management**

The application provides comprehensive status tracking throughout the analysis lifecycle:

-   **QUEUED**: Video uploaded, waiting for processing
-   **PROCESSING**: Active analysis in progress with live updates
-   **ANALYZED**: Complete results available for viewing
-   **FAILED**: Error handling with retry options

Status updates are managed through smart polling with React Query, providing optimal performance and user experience.

## Performance Optimizations

### Code Splitting

-   Route-based code splitting
-   Component lazy loading
-   Dynamic imports for heavy components

### Asset Optimization

-   Image optimization with Cloudinary
-   Video streaming and adaptive quality
-   Bundle size monitoring

### Caching Strategy

-   React Query for server state caching
-   LocalStorage for user preferences
-   Service worker for offline capabilities

## Testing Strategy

### Development Testing

```bash
# Component testing with React Testing Library
npm run test

# E2E testing with Cypress
npm run test:e2e

# Performance testing
npm run test:performance
```

### Quality Assurance

-   ESLint for code quality
-   Prettier for code formatting
-   TypeScript for type safety
-   Accessibility testing with axe-core

## Deployment

### Build Configuration

```bash
# Production build
npm run build

# Build with environment
VITE_ENVIRONMENT=production npm run build

# Analyze bundle size
npm run build:analyze
```

### Deployment Targets

-   **Vercel**: Optimized for React/Vite applications
-   **Netlify**: Continuous deployment with Git integration
-   **AWS S3 + CloudFront**: Scalable static hosting
-   **Docker**: Containerized deployment

## Troubleshooting

### Common Issues

**Build Errors:**

-   Check Node.js version compatibility
-   Verify environment variables
-   Clear node_modules and reinstall

**API Connection Issues:**

-   Verify backend server is running
-   Check CORS configuration
-   Validate environment variables

**Performance Issues:**

-   Analyze bundle size with Vite rollup-plugin-visualizer
-   Optimize images and videos
-   Implement proper code splitting

### Debug Mode

```bash
# Enable debug mode
VITE_DEBUG=true npm run dev

# Enable React DevTools
VITE_ENABLE_DEV_TOOLS=true npm run dev
```

## Contributing

### Development Guidelines

1. Follow React best practices and hooks patterns
2. Use TypeScript for new components
3. Implement proper error boundaries
4. Write unit tests for critical components
5. Follow Tailwind CSS utility-first approach

### Code Style

-   Use functional components with hooks
-   Implement proper prop validation
-   Follow naming conventions (PascalCase for components)
-   Use meaningful variable and function names

## Security Considerations

-   **XSS Protection**: Sanitize user inputs and dynamic content
-   **CSRF Protection**: Implement proper token validation
-   **Content Security Policy**: Configure CSP headers
-   **Secure Storage**: Use secure methods for sensitive data

## License

This project is part of the Drishtiksha AI Deepfake Detection System and is proprietary software developed for educational and research purposes.

---

**Last Updated:** August 16, 2025  
**Version:** 2.0.0 - Production Ready  
**Backend Integration:** ✅ Fully Operational  
**Maintainer:** Drishtiksha AI Team
