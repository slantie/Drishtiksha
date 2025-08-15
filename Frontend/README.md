# Drishtiksha AI - Frontend Module

## Overview

The frontend module is a modern React-based web application for the Drishtiksha AI Deepfake Detection System. It provides an intuitive user interface for video upload, analysis management, and comprehensive deepfake detection results visualization across multiple AI models.

## Tech Stack

### Core Technologies

- **React 19.1.0** - Modern React with latest features and performance optimizations
- **Vite 6.3.5** - Fast build tool and development server
- **React Router DOM 7.6.2** - Client-side routing for SPA navigation
- **TypeScript Support** - Type-safe development environment

### UI/UX Framework

- **Tailwind CSS 3.4.17** - Utility-first CSS framework for rapid UI development
- **Lucide React 0.513.0** - Modern icon library with 1000+ icons
- **Framer Motion 12.23.12** - Production-ready motion library for React
- **Radix UI 1.4.2** - Low-level UI primitives for high-quality design systems
- **Class Variance Authority** - Type-safe variant API for component styling

### State Management & Data Fetching

- **TanStack React Query 5.85.0** - Powerful data synchronization for React
- **React Query DevTools** - Development tools for debugging queries
- **Axios 1.9.0** - Promise-based HTTP client for API communication

### Media & File Handling

- **Cloudinary React 1.14.3** - Cloud-based image and video management
- **React Player 3.3.1** - React component for playing videos
- **Cloudinary Video Player 3.2.0** - Advanced video playback capabilities

### Additional Features

- **React Hot Toast 2.5.2** - Lightweight toast notifications
- **React CountUp 6.5.3** - Animated number counting component
- **HTML2Canvas & jsPDF** - Client-side PDF report generation
- **Prop Types 15.8.1** - Runtime type checking for React props

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

### 🎯 Core Functionality

- **Multi-Model Analysis**: Support for 3 AI models (SIGLIP-LSTM-V1, SIGLIP-LSTM-V3, ColorCues-LSTM-V1)
- **Analysis Types**: Quick, Detailed, Frame-by-Frame, and Visualization analysis
- **Versioning System**: Track multiple analysis versions per video and model
- **Real-time Processing**: Live updates during video analysis with progress indicators

### 🔐 Authentication & Security

- **JWT-based Authentication**: Secure user authentication with token management
- **Protected Routes**: Route-level access control for authenticated users
- **Session Management**: Automatic token refresh and logout handling

### 📊 Data Visualization

- **Interactive Charts**: Real-time visualization of analysis results
- **Progress Tracking**: Visual progress indicators for ongoing analysis
- **Comprehensive Reports**: Detailed analysis breakdowns with charts and metrics

### 🎨 User Experience

- **Responsive Design**: Mobile-first approach with Tailwind CSS
- **Dark/Light Theme**: User-selectable theme preferences
- **Intuitive Navigation**: Clear navigation with React Router
- **Toast Notifications**: User feedback for actions and status updates

### 📱 Media Management

- **Cloudinary Integration**: Cloud-based video storage and delivery
- **Video Player**: Advanced video playback with analysis overlays
- **File Upload**: Drag-and-drop video upload with validation
- **Preview Generation**: Video thumbnails and preview functionality

## Environment Configuration

### Required Environment Variables

Create a `.env` file in the Frontend directory:

```env
# API Configuration
VITE_API_BASE_URL=http://localhost:3000/api
VITE_BACKEND_URL=http://localhost:3000

# Cloudinary Configuration
VITE_CLOUDINARY_CLOUD_NAME=your_cloud_name
VITE_CLOUDINARY_API_KEY=your_api_key

# Application Configuration
VITE_APP_NAME="Drishtiksha AI"
VITE_APP_VERSION="1.0.0"
VITE_ENVIRONMENT="development"

# Feature Flags
VITE_ENABLE_ANALYTICS=true
VITE_ENABLE_DEV_TOOLS=true
```

### Environment-Specific Configurations

**Development:**

- Hot reload enabled
- DevTools available
- Detailed error messages
- Source maps enabled

**Production:**

- Optimized bundles
- Minified assets
- Error boundaries
- Performance monitoring

## Installation & Setup

### Prerequisites

- Node.js 18+ and npm/yarn
- Backend API server running on port 3000
- ML Server running on port 8000

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

### Service Layer Architecture

```javascript
// Example: Video Analysis Service
export const videoService = {
  uploadVideo: (formData) => api.post("/videos/upload", formData),
  getVideoAnalyses: (videoId) => api.get(`/videos/${videoId}/analyses`),
  triggerAnalysis: (videoId, analysisType, model) =>
    api.post(`/videos/${videoId}/analyze`, { analysisType, model }),
  getAnalysisDetails: (analysisId) => api.get(`/analyses/${analysisId}`),
};
```

### React Query Integration

```javascript
// Example: Analysis Data Fetching
export const useVideoAnalyses = (videoId) => {
  return useQuery({
    queryKey: ["videoAnalyses", videoId],
    queryFn: () => videoService.getVideoAnalyses(videoId),
    enabled: !!videoId,
    refetchInterval: 5000, // Real-time updates
  });
};
```

## Component Architecture

### Design System Components

**Base Components:**

- Button variants (primary, secondary, outline, ghost)
- Card layouts (default, hover, interactive)
- Form inputs (text, file, select, textarea)
- Modal dialogs and overlays

**Composite Components:**

- VideoUpload (drag-drop with preview)
- AnalysisCard (result display with actions)
- ProgressIndicator (real-time progress)
- DataTable (sortable, filterable results)

### Page Components

**Results.jsx Features:**

- Model-grouped analysis display
- Latest version filtering
- Interactive analysis cards
- Real-time status updates

**DetailedAnalysis.jsx Features:**

- Comprehensive data visualization
- Frame-by-frame analysis display
- Temporal analysis charts
- Model performance metrics

## Performance Optimizations

### Code Splitting

- Route-based code splitting
- Component lazy loading
- Dynamic imports for heavy components

### Asset Optimization

- Image optimization with Cloudinary
- Video streaming and adaptive quality
- Bundle size monitoring

### Caching Strategy

- React Query for server state caching
- LocalStorage for user preferences
- Service worker for offline capabilities

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

- ESLint for code quality
- Prettier for code formatting
- TypeScript for type safety
- Accessibility testing with axe-core

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

- **Vercel**: Optimized for React/Vite applications
- **Netlify**: Continuous deployment with Git integration
- **AWS S3 + CloudFront**: Scalable static hosting
- **Docker**: Containerized deployment

## Troubleshooting

### Common Issues

**Build Errors:**

- Check Node.js version compatibility
- Verify environment variables
- Clear node_modules and reinstall

**API Connection Issues:**

- Verify backend server is running
- Check CORS configuration
- Validate environment variables

**Performance Issues:**

- Analyze bundle size with Vite rollup-plugin-visualizer
- Optimize images and videos
- Implement proper code splitting

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

- Use functional components with hooks
- Implement proper prop validation
- Follow naming conventions (PascalCase for components)
- Use meaningful variable and function names

## Security Considerations

- **XSS Protection**: Sanitize user inputs and dynamic content
- **CSRF Protection**: Implement proper token validation
- **Content Security Policy**: Configure CSP headers
- **Secure Storage**: Use secure methods for sensitive data

## License

This project is part of the Drishtiksha AI Deepfake Detection System and is proprietary software developed for educational and research purposes.

---

**Last Updated:** August 15, 2025  
**Version:** 1.0.0  
**Maintainer:** Drishtiksha AI Team
